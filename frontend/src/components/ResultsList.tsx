import type { RecommendationItem } from "../api/semantica"

interface Props {
  results: RecommendationItem[]
}

export default function ResultsList({ results }: Props) {
  return (
    <div>
      <h3>Results</h3>
      <ul>
        {results.map((r, i) => (
          <li key={i}>
            <strong>{r.title}</strong> — {r.score.toFixed(3)}
          </li>
        ))}
      </ul>
    </div>
  )
}
