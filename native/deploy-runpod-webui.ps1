<#
Deploy a private Animation Effect native Web UI on one new RunPod CPU pod.
Wrapper around deploy-runpod.ps1 with -ComputeType CPU.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 128)][int]$VcpuCount = 32,
    [ValidateRange(1025, 65535)][int]$LocalPort = 8766,
    [string]$PodName = "animation-effect-webui-$VcpuCount`c",
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\runpod_animation_effect_ed25519'),
    [string]$CredentialStore = 'C:\Workspace_Melbin\_archive\3000\actuators\credstore.py',
    [ValidateRange(60, 900)][int]$ReadyTimeoutSeconds = 300
)

$targetScript = Join-Path $PSScriptRoot 'deploy-runpod.ps1'
& $targetScript -ComputeType CPU -VcpuCount $VcpuCount -LocalPort $LocalPort -PodName $PodName -KeyPath $KeyPath -CredentialStore $CredentialStore -ReadyTimeoutSeconds $ReadyTimeoutSeconds
