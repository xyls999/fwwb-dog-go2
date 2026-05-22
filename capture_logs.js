// 1. 先复制粘贴这段到浏览器控制台（F12 → Console）
// 它会拦截所有 GetTrainLog 请求，把响应存到 window._capturedLogs

const originalFetch = window.fetch;
window._capturedLogs = [];
window._capturedAll = [];

window.fetch = async function(...args) {
  const [url, options] = args;
  const response = await originalFetch.apply(this, args);
  
  if (typeof url === 'string' && url.includes('GetTrainLog')) {
    try {
      const clone = response.clone();
      const data = await clone.json();
      const logs = data?.data?.train_logs || [];
      window._capturedLogs.push(...logs);
      window._capturedAll.push(data);
      console.log(`[Capture] +${logs.length} logs, total=${window._capturedLogs.length}`);
    } catch(e) {
      console.error('[Capture] parse error:', e);
    }
  }
  
  return response;
};

console.log('[Capture] Interceptor installed. Now browse logs on the page; data will be captured automatically.');

// 2. 在页面上正常操作：切换日志分页、筛选模块等
// 等你看完了所有想看的日志，再运行下面这段导出：

function exportLogs() {
  const logs = window._capturedLogs;
  if (logs.length === 0) {
    console.log('No logs captured yet.');
    return;
  }
  const blob = new Blob([logs.map(l => JSON.stringify(l)).join('\n') + '\n'], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `train_logs_${Date.now()}.jsonl`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  console.log(`Exported ${logs.length} logs.`);
}

// 运行 exportLogs() 下载
