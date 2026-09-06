<#
Deploy a private Animation Effect native Web UI on one new RunPod GPU pod.

Prerequisites (one-time):
  - Store the RunPod API key as `runpod/api_key` in the Windows credential store.
  - Create an unencrypted dedicated SSH key pair at the paths below. Do not
    use a project key shared with any other service.

Example (RTX 3090):
  .\deploy-runpod-gpu.ps1 -GpuTypeId 'NVIDIA GeForce RTX 3090' -LocalPort 8766

This script never deletes or stops another pod. Terminate an old pod only
after the new URL has been tested.
#>
[CmdletBinding()]
param(
    [string]$GpuTypeId = 'NVIDIA GeForce RTX 3090',

    [ValidateRange(1025, 65535)]
    [int]$LocalPort = 8766,

    [string]$PodName = "animation-effect-webui-gpu",

    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\runpod_animation_effect_ed25519'),

    [string]$CredentialStore = 'C:\Workspace_Melbin\_archive\3000\actuators\credstore.py',

    [ValidateRange(60, 900)]
    [int]$ReadyTimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$apiBase = 'https://rest.runpod.io/v1'

if (-not (Test-Path -LiteralPath $CredentialStore)) {
    throw "Credential-store script was not found: $CredentialStore"
}
if (-not (Test-Path -LiteralPath $KeyPath) -or -not (Test-Path -LiteralPath "$KeyPath.pub")) {
    throw "Dedicated RunPod key pair was not found at $KeyPath and $KeyPath.pub"
}
if (Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue) {
    throw "Local port $LocalPort is already in use. Choose another -LocalPort."
}

# The secret is captured in memory only: never place it in a command argument,
# file, log, or output object.
$apiKey = (& python $CredentialStore get 'runpod/api_key').Trim()
if (-not $apiKey) { throw 'The credential store returned no RunPod API key.' }
$headers = @{ Authorization = "Bearer $apiKey"; 'Content-Type' = 'application/json'; 'User-Agent' = 'animation-effect-launcher/1.0' }
$publicKey = (Get-Content -LiteralPath "$KeyPath.pub" -Raw).Trim()

$payload = @{
    name                  = $PodName
    imageName             = 'runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04'
    gpuTypeId             = $GpuTypeId
    cloudType             = 'SECURE'
    gpuCount              = 1
    containerDiskInGb     = 20
    volumeInGb            = 0
    ports                 = @('22/tcp')
    env                   = @{ PUBLIC_KEY = $publicKey }
} | ConvertTo-Json -Depth 5

# Exactly one creation request. If capacity is unavailable, the script stops;
# it never silently creates a different-size or fallback pod.
$pod = Invoke-RestMethod -Uri "$apiBase/pods" -Method Post -Headers $headers -Body $payload
$podId = $pod.id
Write-Host "Created RunPod pod $podId; waiting for SSH mapping..."

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
do {
    Start-Sleep -Seconds 5
    $pod = Invoke-RestMethod -Uri "$apiBase/pods/$podId" -Headers $headers
    $podHost = $pod.publicIp
    $sshPort = $pod.portMappings.'22'
} until (($pod.desiredStatus -eq 'RUNNING' -and $podHost -and $sshPort) -or (Get-Date) -ge $deadline)

if (-not $podHost -or -not $sshPort) {
    throw "Pod $podId did not publish SSH before the timeout. It was not deleted."
}

$sshBase = @(
    '-i', $KeyPath, '-p', "$sshPort",
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=3',
    "root@$podHost"
)

# Repository bootstrap is authoritative. It installs OS/Python dependencies and
# clones the current GitHub main branch. Choose an interpreter that actually has
# FastAPI installed (some RunPod images point `pip` and `python3` at different versions).
$remote = @'
set -e
curl -fsSL https://raw.githubusercontent.com/melbinjp/animation_effect/main/native/setup_pod.sh | bash
cd /workspace/animation_effect/native
WEBPY=""
for candidate in python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import fastapi, uvicorn' 2>/dev/null; then
    WEBPY="$candidate"
    break
  fi
done
test -n "$WEBPY"
mkdir -p webui_outputs
nohup "$WEBPY" webui.py --host 127.0.0.1 --port 8765 > webui.log 2>&1 &
echo $! > webui.pid
sleep 3
curl -fsS http://127.0.0.1:8765/ >/dev/null
printf WEBUI_READY
'@

$priorEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$setupOutput = & ssh.exe @sshBase $remote 2>&1
$sshExit = $LASTEXITCODE
$ErrorActionPreference = $priorEAP
if ($sshExit -ne 0 -or "$setupOutput" -notmatch 'WEBUI_READY') {
    throw "Remote setup or Web UI health check failed. Pod $podId remains available for inspection.`n$setupOutput"
}

# Each local tunnel needs its own port; several pods may all use remote 8765.
$tunnelArgs = @(
    '-i', $KeyPath, '-p', "$sshPort", '-N',
    '-L', "127.0.0.1:$LocalPort`:127.0.0.1:8765",
    '-o', 'BatchMode=yes',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=3',
    "root@$podHost"
)
$tunnel = Start-Process -FilePath 'ssh.exe' -ArgumentList $tunnelArgs -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 3
$health = Invoke-WebRequest -Uri "http://127.0.0.1:$LocalPort/" -TimeoutSec 20
if ($health.StatusCode -ne 200 -or $health.Content -notmatch 'Linearty native') {
    throw "Tunnel started but did not serve the expected Web UI at http://127.0.0.1:$LocalPort/"
}

[pscustomobject]@{
    PodId          = $podId
    GpuType        = $pod.gpuTypeId
    PodSshEndpoint = "$podHost`:$sshPort"
    WebUi          = "http://127.0.0.1:$LocalPort/"
    TunnelProcess  = $tunnel.Id
    CostPerHour    = $pod.costPerHr
    OutputFolder   = '/workspace/animation_effect/native/webui_outputs'
}
