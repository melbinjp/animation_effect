<#
Deploy a private Animation Effect native Web UI on one new RunPod GPU pod.
Wrapper around deploy-runpod.ps1 with -ComputeType GPU.
#>
[CmdletBinding()]
param(
    [string]$GpuTypeId = 'NVIDIA GeForce RTX 3090',
    [ValidateRange(1025, 65535)][int]$LocalPort = 8766,
    [string]$PodName = "animation-effect-webui-gpu",
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\runpod_animation_effect_ed25519'),
    [string]$CredentialStore = 'C:\Workspace_Melbin\_archive\3000\actuators\credstore.py',
    [ValidateRange(60, 900)][int]$ReadyTimeoutSeconds = 300
)

$targetScript = Join-Path $PSScriptRoot 'deploy-runpod.ps1'
& $targetScript -ComputeType GPU -GpuTypeId $GpuTypeId -LocalPort $LocalPort -PodName $PodName -KeyPath $KeyPath -CredentialStore $CredentialStore -ReadyTimeoutSeconds $ReadyTimeoutSeconds
