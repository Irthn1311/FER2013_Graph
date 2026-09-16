@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo [RUNNER] FER2013 Hierarchical Pixel-to-Motif Graph Pipeline
echo ============================================================

set CONFIG=standalone\pixel_gnn\configs\fer2013_pixel_motif_graph_fast_seed42.yaml
if not "%~1"=="" set CONFIG=%~1

set OUTPUT_DIR=outputs\pixel_motif_graph
if not "%~2"=="" set OUTPUT_DIR=%~2

set VIS_DIR=%OUTPUT_DIR%\visualizations

echo Config: %CONFIG%
echo Output: %OUTPUT_DIR%
echo Visualizations: %VIS_DIR%

echo.
echo [STEP 1/3] Quick smoke verification...
python train.py --config "%CONFIG%" --smoke --synthetic
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Smoke test failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [STEP 2/3] Launching Training + Validation + Test Pipeline...
python train.py --config "%CONFIG%" --output-root "%OUTPUT_DIR%" %*
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training pipeline failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [STEP 3/3] Generating Test Set Graph Visualizations...
python standalone\pixel_gnn\visualize_test_graph.py --config "%CONFIG%" --checkpoint "%OUTPUT_DIR%\best_val_accuracy.weights.h5" --output-dir "%VIS_DIR%" --num-samples 7
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Visualization script returned error, check logs.
)

echo.
echo ============================================================
echo [COMPLETED] Pipeline finished successfully!
echo Outputs: %OUTPUT_DIR%
echo Visualizations: %VIS_DIR%
echo ============================================================
