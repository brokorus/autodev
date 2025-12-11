[CmdletBinding()]
param(
    [switch]$SkipVenvCreate
)

$parentScript = Resolve-Path (Join-Path $PSScriptRoot "..\\start-autodev.ps1")
& $parentScript @PSBoundParameters
