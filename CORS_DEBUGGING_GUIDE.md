# CORS Debugging Guide

## ❌ Current Error
```
Access to fetch at 'https://scene-forge-backend.onrender.com/processing/files' 
from origin 'https://scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present
```

## ✅ Latest Fix Applied
**Commit**: `29d379d`

Changes:
- Simplified CORS middleware (removed redundant handler)
- Removed custom OPTIONS handler
- Proper middleware ordering
- CORSMiddleware now handles all CORS requests

**Status**: Pushed to GitHub ✅

## 🔴 Next Steps: FORCE RENDER REDEPLOY

The code has been pushed but Render may not have auto-deployed yet.

### Option 1: Manual Render Redeploy (Recommended)
1. Go to https://dashboard.render.com
2. Select **scene-forge-backend** service
3. Click **Manual Deploy** → **Deploy Latest Commit**
4. Wait 2-3 minutes for deployment
5. Try upload again

### Option 2: Check Render Logs
1. Go to Render Dashboard
2. Click **scene-forge-backend**
3. Click **Logs** tab
4. Look for:
   - `CORS Origins: ['http://localhost:3000', ...]` ← Should appear on startup
   - `Incoming: POST /processing/files` ← Should appear on upload
   - Any error messages

### Option 3: Verify CORS is Working (Quick Test)
Open browser console and run:
```javascript
fetch('https://scene-forge-backend.onrender.com/health', {
  method: 'GET',
  headers: { 'Content-Type': 'application/json' }
})
.then(r => r.json())
.then(console.log)
.catch(e => console.error('CORS Error:', e))
```

Expected response:
```
{status: "healthy"}
```

If you get a CORS error:
- Render hasn't redeployed yet
- Or there's a different issue

## 🔍 If Error Persists

### Check 1: Verify Backend URL
The error says: `scene-forge-backend.onrender.com`
This is correct. ✅

### Check 2: Verify Frontend URL
The error says: `scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app`
This is in the CORS allow list. ✅

### Check 3: Browser Cache
Clear browser cache (or use Incognito):
```
Chrome: Cmd+Shift+Delete → Clear "All time"
Firefox: Ctrl+Shift+Delete → Select "Everything"
```

### Check 4: Check Network Tab (DevTools)
1. Open DevTools (F12)
2. Go to Network tab
3. Upload a file
4. Look for the POST request to `/processing/files`
5. Click it, view **Response Headers**
6. Should show:
   ```
   access-control-allow-origin: https://scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app
   access-control-allow-methods: GET, POST, PUT, DELETE, OPTIONS, PATCH
   access-control-allow-headers: *
   ```

If missing: Render is still running old code

## 🆘 Still Not Working?

Try this temporary workaround - allow all origins (NOT for production):

Edit `SceneForge_Backend/app/main.py`:
```python
allowed_origins = ["*"]  # Temporary - allows ANY origin

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # Must be False with allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then:
```bash
git add -A
git commit -m "TEMP: Allow all origins for testing"
git push origin main
```

Wait for Render redeploy, try upload. If it works, CORS is the issue.
If it still fails, the issue is elsewhere (500 error, routing, etc.)

## 📋 Checklist Before Requesting Help

- [ ] Git push was successful (last commit: 29d379d)
- [ ] Render manual redeploy completed (took 2-3 min)
- [ ] Browser cache cleared
- [ ] Checked DevTools Network tab for CORS headers
- [ ] Tested /health endpoint
- [ ] Verified frontend URL is in CORS allow list
- [ ] Verified backend is running (check Render logs)
- [ ] Tried in Incognito/Private mode

## 🎯 Expected Working State

After fix is deployed:
1. Upload button works ✅
2. No CORS error in console ✅
3. 502 error changes to real error (if any processing fails) ✅
4. Response Headers show `access-control-allow-origin` ✅

