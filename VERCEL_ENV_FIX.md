# 🚨 ROOT CAUSE FOUND: Environment Variable Missing on Vercel

## The Problem

Your frontend code defaults to:
```javascript
const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || 
                  process.env.NEXT_PUBLIC_API_BASE || 
                  'http://localhost:8000';  // ← This is the default!
```

**On Vercel**, these env vars are NOT set, so it defaults to `http://localhost:8000`.

But `localhost:8000` doesn't exist in production! This causes:
1. Browser tries to connect to http://localhost:8000 (which doesn't exist)
2. Gets a 502 or connection refused
3. Then CORS blocks it because headers aren't even being returned

## The Solution: Set Environment Variables on Vercel

### Step 1: Go to Vercel Dashboard
https://vercel.com/dashboard

### Step 2: Select **scene-forge** project (frontend)

### Step 3: Click **Settings** → **Environment Variables**

### Step 4: Add New Environment Variable
```
Name:  NEXT_PUBLIC_API_URL
Value: https://scene-forge-backend.onrender.com
```

⚠️ **IMPORTANT**: Must start with `NEXT_PUBLIC_` to be accessible in browser

### Step 5: Click **Add** then **Save**

### Step 6: Redeploy
Click **Deployments** → Click the top deployment → Click **Redeploy**

Wait 2-3 minutes for deployment to complete.

## After Deployment

Your frontend will now use:
```javascript
const API_URL = 'https://scene-forge-backend.onrender.com'
```

Instead of:
```javascript
const API_URL = 'http://localhost:8000' // ❌ Wrong
```

## Verification

After redeployment:
1. Open Vercel app in browser
2. Open DevTools (F12) → Console
3. Look for log:
   ```
   API URL configured as: https://scene-forge-backend.onrender.com
   ```
4. Try uploading a file
5. In Network tab, you should see the POST request go to the correct backend URL
6. CORS error should be gone (or replaced with actual backend error if any)

## Why This Fixes CORS

- ✅ Frontend now requests the correct backend URL
- ✅ Browser no longer tries localhost:8000
- ✅ Options preflight request goes to real backend
- ✅ Real backend returns proper CORS headers
- ✅ Browser allows the request

---

## Alternative: Use vercel.json for Rewrites (Optional)

If you want to proxy requests through Vercel, add to `vercel.json`:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://scene-forge-backend.onrender.com/:path*"
    }
  ]
}
```

Then change frontend to:
```javascript
const API_URL = '/api'
```

This proxies through Vercel, which might be more reliable. But the env var approach is simpler.

---

## Checklist

- [ ] Go to Vercel Dashboard
- [ ] Select scene-forge project
- [ ] Go to Settings → Environment Variables
- [ ] Add: `NEXT_PUBLIC_API_URL = https://scene-forge-backend.onrender.com`
- [ ] Click Save
- [ ] Go to Deployments
- [ ] Click Redeploy on latest commit
- [ ] Wait 2-3 minutes
- [ ] Open browser and check console for correct API URL
- [ ] Try uploading a file
- [ ] CORS error should be gone! ✅
