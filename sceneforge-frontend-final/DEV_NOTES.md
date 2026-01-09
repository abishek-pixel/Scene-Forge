Connecting frontend to backend
=============================

This project ships with a simple, local-only user-context that previously simulated login/signup. I updated the client to use a configurable API base URL so it can call your real backend when available.

Quick summary
- The frontend reads the backend base URL from `NEXT_PUBLIC_API_BASE` (see `.env.local`).
- Default in this repo: `http://localhost:5000`.
- The code attempts fetch requests to endpoints such as `/auth/login`, `/auth/signup`, `/users/:id` and will fall back to the original simulated behavior if the backend is unreachable.

What you (developer) should do
1. Start your backend server (it lives at your path: `D:\Abhishek\Documents\SceneForge_Backend`).
   - Open a PowerShell terminal and run the backend server per your backend instructions (for example, `npm run dev` inside that folder).
2. Confirm which host/port the backend listens on (for example `http://localhost:5000`).
3. Update `.env.local` in this frontend repo to point `NEXT_PUBLIC_API_BASE` to that URL.
   - Example `.env.local` content:

```
NEXT_PUBLIC_API_BASE=http://localhost:5000
```

4. Restart the Next.js dev server (env vars are read at build/dev start):

PowerShell example (from project root):

```
pnpm dev
```

If you use `npm`:

```
npm run dev
```

Notes about endpoints and backend shape
- The frontend currently expects these endpoints (adjust in `lib/user-context.tsx` if your backend differs):
  - POST ${API_BASE}/auth/login -> returns { user, token }
  - POST ${API_BASE}/auth/signup -> returns { user, token }
  - POST ${API_BASE}/auth/logout -> optional, best-effort
  - PATCH ${API_BASE}/users/:id -> returns updated user

If your backend exposes different routes (for example `/api/auth/login`), either change the backend, or update `lib/user-context.tsx` to match the actual paths.

If you want help wiring the exact endpoint paths and payloads, tell me the backend routes and example responses (or share a small snippet of the backend router), and I will adapt the frontend calls precisely.
