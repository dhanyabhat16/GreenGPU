import { useState, useEffect } from 'react'
import { impactSummary } from '../api'
import './SustainabilityImpact.css'

export default function SustainabilityImpact({ profileData, dedupData, evalData }) {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const gpuHours = profileData?.gpu_hours_saved ?? 0
  const testsReduced = dedupData?.result?.reduction_percentage != null
    ? (dedupData.result.reduction_percentage * 100).toFixed(1) + '%'
    : '--'
  const carbonFactor = '0.386 kg CO2/kWh'

  const hasData = profileData || (dedupData?.result) || evalData

  useEffect(() => {
    if (!hasData) {
      setSummary(null)
      return
    }
    let cancelled = false
    setLoading(true)
    impactSummary(profileData, dedupData, evalData)
      .then((data) => {
        if (!cancelled && data?.summary) setSummary(data.summary)
      })
      .catch(() => {
        if (!cancelled) setSummary(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [profileData, dedupData, evalData, hasData])

  return (
    <section className="impact-report">
      <div className="impact-header">
        <span className="impact-icon">🌍</span>
        <h2>SUSTAINABILITY IMPACT REPORT</h2>
      </div>

      <div className="impact-metrics">
        <div className="impact-metric">
          <span className="impact-metric__value">{carbonFactor}</span>
          <span className="impact-metric__label">CARBON FACTOR</span>
        </div>
        <div className="impact-metric">
          <span className="impact-metric__value">{gpuHours.toFixed(3)}h</span>
          <span className="impact-metric__label">GPU HOURS FREED</span>
        </div>
        <div className="impact-metric">
          <span className="impact-metric__value">{testsReduced}</span>
          <span className="impact-metric__label">TESTS REDUCED</span>
        </div>
      </div>

      <div className="impact-summary">
        <h3 className="impact-summary__title">
          <span className="diamond">◆</span> GEMINI AI - IMPACT SUMMARY
        </h3>
        <p className="impact-summary__text">
          {loading
            ? 'Generating AI summary...'
            : summary
              ? summary
              : 'Run profiling and deduplication to generate your sustainability impact summary.'}
        </p>
      </div>
    </section>
  )
}
