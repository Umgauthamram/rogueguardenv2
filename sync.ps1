# Rebuild and Sync script
Write-Host "--- Rebuilding Docker Image: rogueguard2 ---" -ForegroundColor Cyan
docker build -t rogueguard2 .

Write-Host "`n--- Pushing to GitHub and Hugging Face ---" -ForegroundColor Cyan
git add .
git commit -m "sync: enhance reset signature and docs"
git push origin main

Write-Host "`n--- Verification ---" -ForegroundColor Yellow
Write-Host "Direct Space URL: https://gauthamram-rogueguardenv2.hf.space"
Write-Host "Use this URL for submission. Make sure the Space status is 'Running'."

Write-Host "`n--- Sync Complete ---" -ForegroundColor Green
