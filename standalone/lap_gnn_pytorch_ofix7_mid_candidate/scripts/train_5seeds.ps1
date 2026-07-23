param(
  [Parameter(Mandatory=$true)][string]$FerCsv,
  [Parameter(Mandatory=$true)][string]$PriorRoot,
  [Parameter(Mandatory=$true)][string]$OutputRoot,
  [string]$Device = "cuda:0",
  [int]$NumWorkers = 2
)
$ErrorActionPreference = "Stop"
foreach ($Seed in @(42, 1009, 1337, 777, 3407)) {
  & "$PSScriptRoot/train_seed.ps1" -Seed $Seed -FerCsv $FerCsv -PriorRoot $PriorRoot -OutputRoot $OutputRoot -Device $Device -NumWorkers $NumWorkers
}
