package com.hospital.workforce.training;

import jakarta.persistence.*;
import java.time.Instant;
@Entity @Table(name = "training_course") public class TrainingCourse {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @Column(nullable = false, length = 140) private String title;
    @Column(nullable = false) private Instant startAt;
    @Column(nullable = false) private int capacity;
    protected TrainingCourse() { } TrainingCourse(String title, Instant startAt, int capacity) { this.title = title; this.startAt = startAt; this.capacity = capacity; }
    public Long getId() { return id; } public String getTitle() { return title; } public Instant getStartAt() { return startAt; } public int getCapacity() { return capacity; }
}
