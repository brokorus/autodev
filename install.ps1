[CmdletBinding()]
param(
    [string]$Repo = "https://github.com/brokorus/autodev.git",
    [string]$Branch = "main",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Repo = if ($env:AUTODEV_REPO) { $env:AUTODEV_REPO } else { $Repo }
$Branch = if ($env:AUTODEV_BRANCH) { $env:AUTODEV_BRANCH } else { $Branch }

function Write-Info($message) {
    Write-Host "[AutoDev Install] $message" -ForegroundColor Cyan
}

function Ensure-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        throw "Git is required but was not found on PATH. Please install Git and rerun."
    }
    Write-Info "Git found at $($git.Source)"
}

function Confirm-Overwrite {
    param(
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $true
    }

    if ($Force) {
        return $true
    }

    $answer = Read-Host "Found existing $Path. Overwrite? (y/N)"
    if ($answer -match '^(y|yes)$') {
        return $true
    }

    throw "Aborted at user request; leaving $Path unchanged."
}

function Refresh-Autodev {
    param(
        [string]$SourceAutodev,
        [string]$DestAutodev
    )

    $proceed = Confirm-Overwrite -Path $DestAutodev
    if (-not $proceed) { return }

    if (-not (Test-Path $DestAutodev)) {
        New-Item -ItemType Directory -Path $DestAutodev -Force | Out-Null
    }

    # Preserve any existing venv to avoid locked-file issues and speed re-installs.
    Get-ChildItem -Path $DestAutodev -Force | Where-Object { $_.Name -ne "venv" } | ForEach-Object {
        Remove-Item -Recurse -Force -LiteralPath $_.FullName -ErrorAction SilentlyContinue
    }

    Copy-Item -Path (Join-Path $SourceAutodev "*") -Destination $DestAutodev -Recurse -Force
}

$targetRoot = (Get-Location).ProviderPath
$tempDir = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("autodev_" + [guid]::NewGuid().ToString("N"))

Write-Info "Installing AutoDev into $targetRoot"
Write-Info "Using repo $Repo (branch: $Branch)"

Ensure-Git

try {
    git clone --depth 1 --branch $Branch $Repo $tempDir | Out-Null

    $sourceAutodev = Join-Path $tempDir ".autodev"
    if (-not (Test-Path $sourceAutodev)) {
        throw "Cloned repo missing .autodev folder; aborting."
    }

    $destAutodev = Join-Path $targetRoot ".autodev"
    Refresh-Autodev -SourceAutodev $sourceAutodev -DestAutodev $destAutodev

    foreach ($file in @("autodev.ps1", "AUTODEV_README.md")) {
        $sourceFile = Join-Path $tempDir $file
        if (Test-Path $sourceFile) {
            $destFile = Join-Path $targetRoot $file
            $ok = Confirm-Overwrite -Path $destFile
            if ($ok) {
                Copy-Item -Path $sourceFile -Destination $destFile -Force
            }
        }
    }

    Write-Info "AutoDev files copied."
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1) Review .autodev/requirements-autodev.txt if you want to pin/adjust packages."
    Write-Host "2) Run ./autodev.ps1 (PowerShell) from this repo to bootstrap and start AutoDev."
    Write-Host ""
    Write-Host "Tip: re-run this installer later with -Force to refresh AutoDev from $Repo."
}
finally {
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
    }
}
