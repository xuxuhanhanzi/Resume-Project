import http from 'k6/http'
import { check, sleep } from 'k6'

const thresholds = { http_req_failed: ['rate<0.01'], http_req_duration: ['p(95)<300'] }
const smoke = __ENV.K6_SMOKE === 'true'

export const options = smoke
  ? { vus: 5, duration: '15s', thresholds, summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'] }
  : {
      stages: [{ duration: '30s', target: 25 }, { duration: '4m', target: 100 }, { duration: '30s', target: 0 }],
      thresholds,
      summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max']
    }

export default function () {
  const token = __ENV.ACCESS_TOKEN
  const params = { headers: { Authorization: `Bearer ${token}` } }
  const departmentId = __ENV.DEPARTMENT_ID || '1'
  const response = http.get(`${__ENV.BASE_URL || 'http://localhost:8080'}/api/shifts?departmentId=${departmentId}&from=2026-09-01T00:00:00Z&to=2026-10-01T00:00:00Z`, params)
  check(response, { 'shift query succeeded': (r) => r.status === 200 })
  sleep(1)
}
