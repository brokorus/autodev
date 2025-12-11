[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$parentScript = Resolve-Path (Join-Path $PSScriptRoot "..\\autodev.ps1")
& $parentScript @Rest
