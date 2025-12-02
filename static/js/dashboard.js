async function fetchAlerts() {
  try {
    const response = await fetch('/api/alerts');
    if (!response.ok) {
      throw new Error('Failed to fetch alerts');
    }
    const data = await response.json();
    renderAlerts(data.alerts || []);
    document.getElementById('status-indicator').innerText = 'Online';
  } catch (error) {
    console.error(error);
    document.getElementById('status-indicator').innerText = 'Disconnected';
  }
}

function renderAlerts(alerts) {
  const list = document.getElementById('alerts');
  list.innerHTML = '';
  alerts.forEach((alert) => {
    const li = document.createElement('li');
    const priorityClass = `alert-priority-${alert.priority || 'low'}`;
    li.innerHTML = `
      <div class="alert-title ${priorityClass}">${alert.event_type || alert.type}</div>
      <div>${alert.timestamp}</div>
      <div>${alert.description || ''}</div>
    `;
    list.appendChild(li);
  });
}

setInterval(fetchAlerts, 5000);
fetchAlerts();

