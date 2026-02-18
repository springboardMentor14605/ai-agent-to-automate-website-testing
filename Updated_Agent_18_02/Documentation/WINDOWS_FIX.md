# Windows Async Playwright Fix - Technical Details

## Problem
When running async Playwright on Windows with FastAPI/Uvicorn, you may encounter:
```
NotImplementedError
```

This error occurs because Windows uses `WindowsSelectorEventLoop` by default, which doesn't support all the async operations that Playwright requires.

## Root Cause
- **Windows Default**: `WindowsSelectorEventLoop` (limited async support)
- **Playwright Requirement**: `WindowsProactorEventLoopPolicy` (full async support)
- **Conflict**: When FastAPI/Uvicorn creates the event loop, it uses the default policy

## Solution
Set the event loop policy to `WindowsProactorEventLoopPolicy` BEFORE any async operations:

```python
import asyncio
import platform

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

## Implementation

### Files Modified

#### 1. `backend/main.py`
Added at the top of the file (before importing playwright_agent):
```python
import asyncio
import platform

# Fix for Windows: Set event loop policy for async Playwright
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

#### 2. `backend/playwright_agent.py`
Added at the top (before importing PlaywrightExecutor):
```python
import asyncio
import platform

# Fix for Windows: Set event loop policy for async Playwright
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

### Why Both Files?
- **main.py**: When running via FastAPI/Uvicorn
- **playwright_agent.py**: When running standalone for testing

## Testing the Fix

### Quick Test
Run the included test script:
```powershell
cd backend
python test_async_fix.py
```

Expected output:
```
Platform: Windows
Event Loop Policy: WindowsProactorEventLoop Policy
✓ Successfully loaded: Google
✓ Playwright async test passed!
```

### Full Application Test
1. Ensure both servers are running:
   ```powershell
   # Terminal 1 - Backend
   cd backend
   python main.py
   
   # Terminal 2 - Frontend
   cd frontend
   npm run dev
   ```

2. Open http://localhost:5173

3. Test with these credentials:
   - URL: `https://www.saucedemo.com/`
   - Username: `standard_user`
   - Password: `secret_sauce`

4. Expected result: Test should complete successfully with "PASS" status

## Technical Details

### Event Loop Policies Comparison

| Policy | Windows Support | Playwright Compatible | Performance |
|--------|----------------|----------------------|-------------|
| WindowsSelectorEventLoop | Default | ❌ No | Limited |
| WindowsProactorEventLoopPolicy | Available | ✅ Yes | Full |

### What ProactorEventLoop Enables
- **I/O Completion Ports**: Windows-native async I/O
- **Subprocesses**: Full subprocess support in async
- **Sockets**: Enhanced socket operations
- **File I/O**: Asynchronous file operations

### Why Playwright Needs It
Playwright's async API relies on:
1. WebSocket connections (browser communication)
2. Subprocess management (browser process)
3. File I/O operations (screenshots, downloads)
4. Network operations (CDP protocol)

All these require ProactorEventLoop on Windows.

## Alternative Approaches (Not Recommended)

### 1. Use Sync API
❌ **Don't do this**: It conflicts with FastAPI's async nature
```python
# BAD - causes blocking
from playwright.sync_api import sync_playwright
```

### 2. Run in Thread Pool
❌ **Don't do this**: Adds complexity and overhead
```python
# BAD - unnecessary complexity
await run_in_executor(ThreadPoolExecutor(), sync_function)
```

### 3. Use Different Event Loop
❌ **Don't do this**: May break other Windows features
```python
# BAD - not cross-platform
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
```

## Best Practices

### ✅ Do This
1. Set policy at application startup (before any async operations)
2. Use platform check for cross-platform compatibility
3. Document the requirement in README/setup instructions

### ❌ Don't Do This
1. Set policy in the middle of execution
2. Change policy multiple times
3. Forget to document Windows requirements

## Verification Checklist

After implementing the fix, verify:
- [ ] `test_async_fix.py` runs without errors
- [ ] Backend server starts successfully
- [ ] Test execution completes (PASS or FAIL, not error)
- [ ] Screenshots are captured
- [ ] Browser launches and closes properly
- [ ] No `NotImplementedError` in logs

## Common Mistakes

### 1. Setting Policy Too Late
```python
# WRONG - agent already imported
from playwright_agent import app
asyncio.set_event_loop_policy(...)  # Too late!
```

### 2. Only Setting in One File
```python
# WRONG - only in main.py
# playwright_agent.py will fail when run standalone
```

### 3. Hardcoding Windows Policy
```python
# WRONG - breaks on Linux/Mac
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

## Related Issues

This fix also resolves:
- `RuntimeError: Event loop is closed`
- `RuntimeError: There is no current event loop`
- Browser hang/freeze on Windows
- Timeout errors specific to Windows

## Further Reading
- [Python asyncio Event Loops](https://docs.python.org/3/library/asyncio-eventloop.html)
- [Playwright Python Async API](https://playwright.dev/python/docs/api/class-playwright)
- [Windows Proactor Event Loop](https://docs.python.org/3/library/asyncio-platforms.html#windows)

---
**Status**: ✅ Implemented and Tested  
**Platform**: Windows 10/11  
**Python**: 3.8+  
**Last Updated**: February 2026
