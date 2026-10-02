<#
Manage RunPod cloud instances and tunnels for Linearty Studio.

Usage:
  # List all active pods and costs:
  .\manage-runpod.ps1 -List

  # Terminate a pod and close any local SSH tunnels forwarding to it:
  .\manage-runpod.ps1 -Terminate 'pod-id'

  # Close all background SSH tunnels listening on port 8765/8766:
  .\manage-runpod.ps1 -CloseTunnels
#>
[CmdletBinding(DefaultParameterSetName = 'List')]
param(
    [Parameter(ParameterSetName = 'List')]
    [switch]$List,

    [Parameter(ParameterSetName = 'Terminate', Mandatory = $true)]
    [string]$Terminate,

    [Parameter(ParameterSetName = 'CloseTunnels')]
    [switch]$CloseTunnels,

    [string]$CredentialStore = 'C:\Workspace_Melbin\_archive\3000\actuators\credstore.py'
)

$ErrorActionPreference = 'Stop'
$apiBase = 'https://rest.runpod.io/v1'

function Get-ApiKey {
    if (-not (Test-Path -LiteralPath $CredentialStore)) {
        throw "Credential store script was not found: $CredentialStore"
    }
    $key = (& python $CredentialStore get 'runpod/api_key').Trim()
    if (-not $key) { throw 'No RunPod API key returned from credential store.' }
    return $key
}

function Stop-LocalTunnels {
    param([int[]]$Ports = @(8765, 8766))
    $stopped = 0
    Get-Process -Name 'ssh' -ErrorAction SilentlyContinue | ForEach-Object {
        $p = $_
        try {
            $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($p.Id)").CommandLine
            foreach ($port in $Ports) {
                if ($cmd -match "127\.0\.0\.1:$port") {
                    Write-Host "Stopping SSH tunnel process $($p.Id) on port $port..."
                    Stop-Process -Id $p.Id -Force
                    $stopped++
                    break
                }
            }
        } catch {}
    }
    if ($stopped -eq 0) {
        Write-Host "No active SSH tunnel processes found on ports $($Ports -join ', ')."
    } else {
        Write-Host "Closed $stopped SSH tunnel process(es)."
    }
}

if ($CloseTunnels) {
    Stop-LocalTunnels
    exit 0
}

$apiKey = Get-ApiKey
$headers = @{
    Authorization = "Bearer $apiKey"
    'Content-Type' = 'application/json'
    'User-Agent'   = 'animation-effect-launcher/1.0'
}

if ($Terminate) {
    Write-Host "Terminating RunPod pod $Terminate..."
    try {
        $res = Invoke-RestMethod -Uri "$apiBase/pods/$Terminate" -Method Delete -Headers $headers
        Write-Host "Pod $Terminate terminated successfully."
    } catch {
        Write-Warning "Could not terminate pod $Terminate via API: $_"
    }
    Stop-LocalTunnels
    exit 0
}

# Default: List pods
Write-Host "Querying RunPod active instances..."
try {
    $pods = Invoke-RestMethod -Uri "$apiBase/pods" -Method Get -Headers $headers
} catch {
    throw "Failed to query RunPod API: $_"
}

if (-not $pods -or $pods.Count -eq 0) {
    Write-Host "No active RunPod pods found. You are not incurring any cloud billing."
    exit 0
}

Write-Host "Found $($pods.Count) active pod(s):`n"
$pods | ForEach-Object {
    [pscustomobject]@{
        PodId        = $_.id
        Name         = $_.name
        Status       = $_.desiredStatus
        Type         = if ($_.gpuTypeId) { $_.gpuTypeId } else { "$($_.vcpuCount) vCPUs" }
        CostPerHour  = "`$$($_.costPerHr)/hr"
        PublicIp     = $_.publicIp
        SshPort      = $_.portMappings.'22'
    }
} | Format-Table -AutoSize

Write-Host "`nTo terminate a pod and close its tunnel when finished:"
Write-Host "  .\manage-runpod.ps1 -Terminate '<PodId>'"
