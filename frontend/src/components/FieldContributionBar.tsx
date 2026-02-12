import { motion } from "framer-motion"

interface Props {
  label: string
  value: number
}

export default function FieldContributionBar({ label, value }: Props) {
  const widthPercent = value * 100

  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>

      <div className="bar-track">
        <motion.div
          className="bar-fill"
          initial={{ width: 0 }}
          animate={{ width: `${widthPercent}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </div>

      <span className="bar-value">{value.toFixed(3)}</span>
    </div>
  )
}