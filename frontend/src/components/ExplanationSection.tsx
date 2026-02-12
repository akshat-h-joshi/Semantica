import FieldContributionBar from "./FieldContributionBar"
import type { Explanation } from "../api/semantica"

interface Props {
  explanation: Explanation
  depth?: number
}

export default function ExplanationSection({
  explanation,
  depth = 0,
}: Props) {
  return (
    <div className="explanation-section" style={{ marginLeft: depth * 16 }}>
      <h4 className="explanation-model">
        {explanation.model.toUpperCase()}
      </h4>

      <p className="explanation-reason">
        {explanation.reason}
      </p>

      {/* Similarity fields */}
      {explanation.fields && (
        <>
          <h5>Similarity contribution</h5>
          {Object.entries(explanation.fields).map(([field, value]) => (
            <FieldContributionBar
              key={field}
              label={field}
              value={value}
            />
          ))}
        </>
      )}

      {/* TF-IDF matched terms */}
      {explanation.matched_terms && (
        <>
          <h5>Matched terms</h5>
          {Object.entries(explanation.matched_terms).map(
            ([source, terms]) => (
              <div key={source} className="matched-terms">
                <strong>{source}</strong>: {terms.join(", ")}
              </div>
            )
          )}
        </>
      )}

      {/* Hybrid components */}
      {explanation.components && (
        <div className="explanation-components">
          {Object.entries(explanation.components).map(
            ([name, component]) => (
              <ExplanationSection
                key={name}
                explanation={component}
                depth={depth + 1}
              />
            )
          )}
        </div>
      )}
    </div>
  )
}