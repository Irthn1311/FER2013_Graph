param(
  [Parameter(Mandatory=$true)][int]$Seed,
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [Parameter(Mandatory=$true)][string]$FerCsv,
  [Parameter(Mandatory=$true)][string]$PriorRoot,
  [string]$Device = "cuda:0"
)
$ErrorActionPreference = "Stop"
python -m lap_gnn.cli.evaluate --config "configs/fer2013_ofix7_mid_seed$Seed.yaml" --checkpoint $Checkpoint --fer-csv $FerCsv --prior-root $PriorRoot --split test --device $Device --num-workers 0
