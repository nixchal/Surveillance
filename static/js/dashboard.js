document.addEventListener('DOMContentLoaded', () => {
    const alertsList = document.getElementById('alerts-list');
    const alertCount = document.getElementById('alert-count');
    const clock = document.getElementById('clock');
    
    // Clock
    setInterval(() => {
        const now = new Date();
        clock.textContent = now.toLocaleTimeString();
    }, 1000);

    // Fetch Alerts
    async function fetchAlerts() {
        try {
            const response = await fetch('/api/alerts');
            const data = await response.json();
            updateAlerts(data.alerts);
        } catch (error) {
            console.error('Error fetching alerts:', error);
        }
    }

    function updateAlerts(alerts) {
        if (!alerts || alerts.length === 0) {
            alertsList.innerHTML = '<div class="empty-state">No active alerts</div>';
            alertCount.textContent = '0';
            return;
        }

        alertCount.textContent = alerts.length;
        alertsList.innerHTML = alerts.map(alert => createAlertHTML(alert)).join('');
        
        // Re-initialize icons for new content
        if (window.lucide) {
            lucide.createIcons();
        }
    }

    function createAlertHTML(alert) {
        const priorityClass = alert.priority ? alert.priority.toLowerCase() : 'medium';
        const iconName = getIconForType(alert.event_type);
        
        // Format timestamp
        const time = new Date(alert.timestamp).toLocaleTimeString();

        return `
            <div class="alert-item ${priorityClass}">
                <div class="alert-icon">
                    <i data-lucide="${iconName}"></i>
                </div>
                <div class="alert-content">
                    <div class="alert-title">
                        ${alert.event_type}
                        <span class="alert-time">${time}</span>
                    </div>
                    <div class="alert-desc">${alert.description}</div>
                </div>
            </div>
        `;
    }

    function getIconForType(type) {
        type = type.toLowerCase();
        if (type.includes('fire')) return 'flame';
        if (type.includes('fight')) return 'swords'; // or 'activity'
        if (type.includes('smoke')) return 'cigarette'; // might need custom or fallback
        if (type.includes('weapon')) return 'crosshair';
        if (type.includes('zone')) return 'ban';
        if (type.includes('crowd')) return 'users';
        return 'alert-triangle';
    }

    // Poll for alerts every 2 seconds
    setInterval(fetchAlerts, 2000);
    fetchAlerts();
});
