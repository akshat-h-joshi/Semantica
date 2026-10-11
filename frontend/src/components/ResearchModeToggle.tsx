import "../App.css"

interface Props {
  researchMode: boolean
  onToggle: () => void
}

export default function ResearchModeToggle({ researchMode, onToggle }: Props) {
  return (
    <button
      className={`research-toggle__pill${researchMode ? " research-toggle__pill--on" : ""}`}
      role="switch"
      aria-checked={researchMode}
      aria-label="Toggle research mode"
      onClick={onToggle}
      title="Research mode"
    >
      <span className="research-toggle__thumb" />
      <span className="research-toggle__text">
        {researchMode ? "Research" : "Research"}
      </span>
    </button>
  )
}