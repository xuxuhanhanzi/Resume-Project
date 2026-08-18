package com.hospital.workforce.notification;

import com.hospital.workforce.scheduling.ShiftPublishedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
@Component class NotificationListener {
    private final NotificationRepository notifications;
    NotificationListener(NotificationRepository notifications) { this.notifications = notifications; }
    @EventListener void on(ShiftPublishedEvent event) { notifications.save(new Notification(event.employeeId(), "SHIFT_PUBLISHED", "A shift has been published: " + event.shiftId())); }
}
