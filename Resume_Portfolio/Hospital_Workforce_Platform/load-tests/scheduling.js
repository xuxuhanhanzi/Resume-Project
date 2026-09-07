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

export function setup() {
  if (__ENV.ACCESS_TOKEN) return { token: __ENV.ACCESS_TOKEN }
  const response = http.post(
    `${__ENV.KEYCLOAK_URL || 'http://keycloak:8080'}/realms/hospital/protocol/openid-connect/token`,
    {
      client_id: 'hospital-api',
      grant_type: 'password',
      username: 'hr.admin',
      password: __ENV.DEMO_PASSWORD || 'Password1!'
    }
  )
  check(response, { 'Keycloak login succeeded': (r) => r.status === 200 })
  if (response.status !== 200) throw new Error(`Keycloak login failed: ${response.status}`)
  return { token: response.json('access_token') }
}

export default function (data) {
  const token = data.token
  const params = { headers: { Authorization: `Bearer ${token}` } }
  const departmentId = __ENV.DEPARTMENT_ID || '1'
  const response = http.get(`${__ENV.BASE_URL || 'http://localhost:8080'}/api/shifts?departmentId=${departmentId}&from=2026-09-01T00:00:00Z&to=2026-10-01T00:00:00Z`, params)
  check(response, { 'shift query succeeded': (r) => r.status === 200 })
  sleep(1)
}
