# 🎯 Backend Configuration Quick Reference

## 📋 Three Ways to Configure Backend

### 1️⃣ Settings UI (Recommended) ⭐
**Best for:** Easy switching between local and remote

```
1. Open Nexus → Settings (⚙️)
2. Scroll to "Backend Connection"
3. Enter URL or use preset
4. Click "Save"
5. Refresh page (F5)
```

**Pros:** No code changes, no restart needed (just refresh)  
**Cons:** None

---

### 2️⃣ Environment Variable
**Best for:** Permanent custom setup

Create `src/frontend/.env.local`:
```env
NEXT_PUBLIC_BACKEND_URL=http://your-custom-url:8000
```

Then restart frontend:
```powershell
npm run dev
```

**Pros:** Set once, forget it  
**Cons:** Requires restart to change

---

### 3️⃣ Code Helper (For Developers)
**Best for:** Custom integrations

```typescript
import { getBackendUrl, fetchFromBackend } from '@/lib/backend-config';

// Get current URL
const url = getBackendUrl();

// Make API call
const response = await fetchFromBackend('/models/available');
```

**Pros:** Flexible, programmatic control  
**Cons:** Requires code changes

---

## 🌐 Common URLs

| Type | URL | Use Case |
|------|-----|----------|
| Local (default) | `http://localhost:8000` | Development |
| Local (alt port) | `http://localhost:8080` | Port conflict |
| Network | `http://192.168.1.X:8000` | Access from other devices |
| Custom | `http://your-ip:8000` | Remote server |

---

## ✅ Quick Test

Test your backend URL:

### In Browser:
```
https://your-url/health
```

Should return:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-12T...",
  "ollama_running": true
}
```

### In Settings UI:
1. Enter URL
2. Click "Test" button
3. Look for green checkmark ✅

---

## 🔄 Switching Backends

### Local → Network/Remote
1. Start backend on remote machine or find your network IP
2. Enter URL (e.g., `http://192.168.1.100:8000`) in Settings UI
3. Click Save & Refresh

### Network/Remote → Local
1. Start local backend:
   ```powershell
   python scripts\launch.py --backend-only
   ```
2. Click "Local (Port 8000)" preset in Settings
3. Click Save & Refresh

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't connect | Check URL has no trailing slash |
| 404 Not Found | Verify backend is running |
| CORS error | Backend must allow frontend origin |
| Stale URL | Clear cache (Ctrl+Shift+Del) and refresh |
| Changes not applying | Hard refresh (Ctrl+F5) |

---

## 📞 Quick Help

**Backend not running?**
```powershell
# Start local backend
python scripts\launch.py --backend-only

# Check if running
curl http://localhost:8000/health
```

**Settings not saving?**
```javascript
// In browser console
localStorage.getItem('nexus_backend_url')  // Check current
localStorage.removeItem('nexus_backend_url')  // Reset
```

**Need API docs?**
```
Visit: http://your-backend-url/docs
```

---

**Made with ❤️ by Nexus Team**
