param(
  [Parameter(Mandatory=$true)][int]$Seed,
  [Parameter(Mandatory=$true)][string]$FerCsv,
  [Parameter(Mandatory=$true)][string]$PriorRoot,
  [Parameter(Mandatory=$true)][string]$OutputRoot,
  [string]$Device = "cuda:0",
  [int]$NumWorkers = 2
)
$ErrorActionPreference = "Stop"
python -m lap_gnn.cli.train --config "configs/fer2013_ofix7_mid_seed$Seed.yaml" --fer-csv $FerCsv --prior-root $PriorRoot --output-root $OutputRoot --device $Device --num-workers $NumWorkers --no-resume
