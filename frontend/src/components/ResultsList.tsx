import { useState } from "react"
import type { RecommendationItem } from "../api/semantica"
import { motion } from "framer-motion"
import HighlightedAbstract  from "./HighlightedAbstract"
import ExplanationModal from "./ExplanationModal"

interface Props {
  results: RecommendationItem[]
}

export default function ResultsList({ results }: Props) {
  const [openExplanation, setOpenExplanation] =
    useState<RecommendationItem | null>(null)

  return (
    <>
      <div className="results-grid">
        {results.map((item, idx) => (
          <motion.div
            key={idx}
            className="result-card"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
          >
            <h3>
              <motion.a
                href={item.link}
                target="_blank"
                rel="noreferrer"
                className="paper-title"
              >
                {item.title}
              </motion.a>
            </h3>

            {/* <p className="score">Score: {item.score.toFixed(3)}</p> */}

            <div className="abstract-container">
              <HighlightedAbstract
                abstract={item.abstract}
                keywords={item.keywords}
              />
            </div>

            <button
              className="explain-btn"
              onClick={() => setOpenExplanation(item)}
            >
              Why recommend?
            </button>
          </motion.div>
        ))}
      </div>

      {openExplanation && (
        <ExplanationModal
          explanation={openExplanation.explanation}
          onClose={() => setOpenExplanation(null)}
        />
      )}
    </>
  )
}
