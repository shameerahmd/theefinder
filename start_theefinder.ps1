$ErrorActionPreference = "Continue"

$root = "D:\New folder (6)\theefinder"
$frontend = Join-Path $root "frontend"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "          STARTING THEEFINDER" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --------------------------------------------------
# 1. POSTGIS
# --------------------------------------------------

Write-Host "`n[1/4] Starting PostGIS..." -ForegroundColor Yellow

Set-Location $root
docker compose up -d

Start-Sleep -Seconds 4

# --------------------------------------------------
# 2. FASTAPI
# --------------------------------------------------

Write-Host "`n[2/4] Starting FastAPI..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root'; uv run --project backend fastapi dev backend\app\main.py"
)

Write-Host "Waiting for FastAPI..." -ForegroundColor Gray

$backendReady = $false

for ($i = 1; $i -le 40; $i++) {

    try {

        $health = Invoke-RestMethod `
            -Uri "http://127.0.0.1:8000/health" `
            -TimeoutSec 3

        if ($health.status -eq "healthy") {

            $backendReady = $true
            Write-Host "FastAPI READY" -ForegroundColor Green
            break
        }

    }
    catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $backendReady) {

    Write-Host "FastAPI did not start. Cloudflare will NOT be started." -ForegroundColor Red
    exit
}

# --------------------------------------------------
# 3. NEXT.JS PRODUCTION
# --------------------------------------------------

Write-Host "`n[3/4] Starting Next.js..." -ForegroundColor Yellow

Set-Location $frontend

if (-not (Test-Path ".next")) {

    Write-Host "Production build not found. Building..." -ForegroundColor Yellow

    npm run build

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Next.js build FAILED." -ForegroundColor Red
        exit
    }
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$frontend'; npm run start -- --hostname 0.0.0.0"
)

Write-Host "Waiting for Next.js..." -ForegroundColor Gray

$frontendReady = $false

for ($i = 1; $i -le 40; $i++) {

    try {

        $response = Invoke-WebRequest `
            -Uri "http://127.0.0.1:3000" `
            -UseBasicParsing `
            -TimeoutSec 3

        if ($response.StatusCode -eq 200) {

            $frontendReady = $true
            Write-Host "Next.js READY" -ForegroundColor Green
            break
        }

    }
    catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $frontendReady) {

    Write-Host "Next.js did not start. Cloudflare will NOT be started." -ForegroundColor Red
    exit
}

# --------------------------------------------------
# TEST FRONTEND -> BACKEND
# --------------------------------------------------

Write-Host "`nTesting TheeFinder API..." -ForegroundColor Yellow

try {

    $test = Invoke-RestMethod `
        -Uri "http://127.0.0.1:3000/api/chennai?days=5&max_detections=25" `
        -TimeoutSec 120

    Write-Host "FIRMS      :" $test.firms_detection_count -ForegroundColor Green
    Write-Host "Classified :" $test.classified_count -ForegroundColor Green
    Write-Host "Failed     :" $test.failed_count -ForegroundColor Green

}
catch {

    Write-Host "TheeFinder API test failed." -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit
}

# --------------------------------------------------
# 4. CLOUDFLARE
# --------------------------------------------------

Write-Host "`n[4/4] Starting Cloudflare Tunnel..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cloudflared tunnel --url http://127.0.0.1:3000"
)

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "       THEEFINDER IS RUNNING" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Local Dashboard : http://localhost:3000"
Write-Host "FastAPI          : http://127.0.0.1:8000"
Write-Host "API Docs         : http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host "PUBLIC URL:" -ForegroundColor Cyan
Write-Host "Look at the Cloudflare window for the new" -ForegroundColor Cyan
Write-Host "https://xxxxx.trycloudflare.com address." -ForegroundColor Cyan
Write-Host ""