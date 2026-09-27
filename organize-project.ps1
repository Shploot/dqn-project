# organize-project.ps1
#
# Run this from the folder containing ALL your project files, flat
# (both your original PC files and whatever you just copied over from
# the Pi). It creates the folder structure and moves known files into
# place. Anything it doesn't recognize is left alone and listed at the
# end, so you can handle stragglers by hand.
#
# Usage: just run it from that folder:
#   .\organize-project.ps1

$folders = @(
    "simulation",
    "hardware",
    "hardware\calibration",
    "models",
    "logs",
    "analysis",
    "paper"
)

foreach ($f in $folders) {
    New-Item -ItemType Directory -Force -Path $f | Out-Null
}

function Move-IfExists($file, $dest) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination $dest -Force
        Write-Host "  Moved: $file -> $dest"
        return $true
    }
    return $false
}

Write-Host "`n=== Simulation (PC-side) ===" -ForegroundColor Cyan
$simulation_files = @(
    "nav_env.py", "dqn_agent.py", "train_corridor.py", "train_ppo.py",
    "astar_planner.py", "compare.py", "train.py"
)
foreach ($f in $simulation_files) { Move-IfExists $f "simulation" | Out-Null }

Write-Host "`n=== Hardware (Pi-side) ===" -ForegroundColor Cyan
$hardware_files = @(
    "deploy.py", "deploy_astar.py", "odometry.py", "reactive_avoid.py"
)
foreach ($f in $hardware_files) { Move-IfExists $f "hardware" | Out-Null }

Write-Host "`n=== Calibration/diagnostic scripts ===" -ForegroundColor Cyan
$calibration_files = @(
    "sensor_test.py", "motor_test.py", "move_test.py", "rotate90_test.py",
    "forward_drift_test.py", "turn_drift_test.py", "single_turn_test.py",
    "forward_then_turn.py", "gpio_check.py"
)
foreach ($f in $calibration_files) { Move-IfExists $f "hardware\calibration" | Out-Null }

Write-Host "`n=== Trained models ===" -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "*.pt" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "models" -Force
    Write-Host "  Moved: $($_.Name) -> models"
}
Get-ChildItem -Path . -Filter "*.zip" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "models" -Force
    Write-Host "  Moved: $($_.Name) -> models"
}

Write-Host "`n=== Trial logs ===" -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "*.log" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "logs" -Force
    Write-Host "  Moved: $($_.Name) -> logs"
}

Write-Host "`n=== Analysis tools and outputs ===" -ForegroundColor Cyan
$analysis_files = @("analyze_run.py", "compare_experiments.py")
foreach ($f in $analysis_files) { Move-IfExists $f "analysis" | Out-Null }
Get-ChildItem -Path . -Filter "*.csv" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "analysis" -Force
    Write-Host "  Moved: $($_.Name) -> analysis"
}
Get-ChildItem -Path . -Filter "comparison_charts*.png" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "analysis" -Force
    Write-Host "  Moved: $($_.Name) -> analysis"
}
Get-ChildItem -Path . -Filter "training_progress*.png" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "analysis" -Force
    Write-Host "  Moved: $($_.Name) -> analysis"
}

Write-Host "`n=== Paper ===" -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "*.docx" -File | ForEach-Object {
    Move-Item -Path $_.FullName -Destination "paper" -Force
    Write-Host "  Moved: $($_.Name) -> paper"
}

Write-Host "`n=== Done. Anything left in this folder wasn't recognized: ===" -ForegroundColor Yellow
Get-ChildItem -Path . -File | ForEach-Object { Write-Host "  $($_.Name)" }
