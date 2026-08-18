<script setup>
import { ref } from 'vue'

const sections = [
  ['组织与人员', '维护科室、员工、岗位及资质。'],
  ['并发安全排班', '通过员工行锁、重叠校验和幂等键阻止重复班次。'],
  ['薪资审批', '以不可变快照保存计算结果，并按状态机审批。'],
  ['审计与监控', '查看操作记录、指标与性能报告。']
]
const active = ref(0)
</script>

<template>
  <main>
    <header><p class="eyebrow">Hospital Workforce Platform</p><h1>医院人员与排班管理平台</h1><p>模块化单体 · Java 21 · Spring Boot · MySQL · Keycloak</p></header>
    <section class="cards"><button v-for="([title, text], index) in sections" :key="title" :class="{ active: active === index }" @click="active = index"><strong>{{ title }}</strong><span>{{ text }}</span></button></section>
    <section class="panel"><h2>{{ sections[active][0] }}</h2><p>{{ sections[active][1] }}</p><p>后端 API 文档：<a href="http://localhost:8080/swagger-ui/index.html" target="_blank">Swagger UI</a></p><p>开发环境登录由 Keycloak 提供；演示账号和启动方法见项目 README。</p></section>
  </main>
</template>
