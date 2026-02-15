import './MetricCards.css'

export default function MetricCards({ profileData, dedupData, evalData }) {
  const stats = profileData?.stats
  const dedup = dedupData?.result
  const ev = evalData

  const gpuHoursSaved = profileData?.gpu_hours_saved ?? 0
  const energyKwh = profileData?.energy_saved_kwh ?? 0
  const carbonKg = profileData?.carbon_saved_kg ?? 0
  const costUsd = profileData?.cost_saved_usd ?? 0

  const cards = [
    {
      icon: '⚡',
      value: `${gpuHoursSaved.toFixed(4)}h`,
      label: 'GPU HOURS SAVED',
      color: 'orange',
    },
    {
      icon: '🔋',
      value: `${energyKwh.toFixed(4)} kWh`,
      label: 'ENERGY SAVED',
      color: 'cyan',
    },
    {
      icon: '🌿',
      value: `${carbonKg.toFixed(4)} kg`,
      label: 'CO2 AVOIDED',
      color: 'green',
    },
    {
      icon: '💰',
      value: `$${costUsd.toFixed(2)} USD`,
      label: 'COST SAVED',
      color: 'yellow',
    },
  ]

  return (
    <div className="metric-cards">
      {cards.map((card) => (
        <div key={card.label} className={`metric-card metric-card--${card.color}`}>
          <span className="metric-card__icon">{card.icon}</span>
          <span className="metric-card__value">{card.value}</span>
          <span className="metric-card__label">{card.label}</span>
        </div>
      ))}
    </div>
  )
}
