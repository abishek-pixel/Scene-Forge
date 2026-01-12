# 🎯 CORS Error - Root Cause Confirmed

## What You Found
When you click: `https://scene-forge-backend.onrender.com/processing/files`

You get:
```json
{"detail":"Method Not Allowed"}
```

## Why This Happens
- ✅ Your backend is running and responding
- ✅ The endpoint exists and is configured correctly
- ✅ It's correctly rejecting GET requests
- ❌ But you're doing a GET (by clicking the link)
- ❌ The endpoint requires POST (for file uploads)

## The Real Issue: Environment Variables Not Set

Your frontend is **NOT using the correct backend URL** because the environment variable is missing on Vercel.

**Current Flow (BROKEN):**
```
Frontend (Vercel) 
  → No env var set
  → Defaults to http://localhost:8000
  → Can't reach localhost in production
  → Tries anyway
  → Browser blocks with CORS error
```

**Required Flow (WORKING):**
```
Frontend (Vercel)
  → NEXT_PUBLIC_API_URL = https://scene-forge-backend.onrender.com
  → Makes POST request to real backend
  → Backend returns proper CORS headers
  → Browser allows request
  → Upload works ✅
```

## IMMEDIATE ACTION NEEDED

### 1. Go to Vercel Dashboard
https://vercel.com/dashboard

### 2. Click **scene-forge** project

### 3. Click **Settings**

### 4. Click **Environment Variables**

### 5. Click **Add New**

Fill in:
- **Name:** `NEXT_PUBLIC_API_URL`
- **Value:** `https://scene-forge-backend.onrender.com`
- **Environments:** Check all (Production, Preview, Development)

### 6. Click **Add** then **Save**

### 7. Go to **Deployments**

### 8. Click your latest deployment

### 9. Click **Redeploy** button

### 10. Wait 2-3 minutes for redeploy

### 11. Test Upload

Open your frontend:
```
https://scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app
```

Open DevTools (F12) → Console

You should see:
```
API URL configured as: https://scene-forge-backend.onrender.com
```

NOT:
```
API URL configured as: http://localhost:8000
```

Then try uploading a file. It should work now!

---

## Testing the Endpoint Directly (For Reference)

The endpoint requires:
- **Method:** POST (not GET)
- **Body:** FormData with files and scene_name
- **CORS Header:** Access-Control-Allow-Origin must be returned

Example (using curl in PowerShell):
```powershell
$filePath = "C:\path\to\test.jpg"
$file = Get-Item $filePath
$fileStream = $file.OpenRead()

$form = @{
    files = $fileStream
    scene_name = "test_scene"
}

$response = Invoke-WebRequest `
  -Uri "https://scene-forge-backend.onrender.com/processing/files" `
  -Method POST `
  -Form $form

$response | ConvertTo-Json
```

But **don't do this manually** - just set the env var and let the frontend UI handle it!

---

## Summary

| What | Status | Fix |
|------|--------|-----|
| Backend endpoint | ✅ Working (responds to POST) | None needed |
| CORS configuration | ✅ Set correctly on backend | None needed |
| Environment variable on Vercel | ❌ Missing | **ADD IT NOW** |
| Frontend using correct URL | ❌ Using localhost:8000 | Set env var to fix |

**Next step:** Add `NEXT_PUBLIC_API_URL` env var on Vercel, redeploy, then try uploading.
