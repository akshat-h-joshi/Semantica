import { motion } from "framer-motion"
import { createPortal } from "react-dom"
import type { Explanation } from "../api/semantica"
import ExplanationSection from "./ExplanationSection"

interface Props {
  explanation: Explanation
  onClose: () => void
}

export default function ExplanationModal({ explanation, onClose }: Props) {
  return createPortal(
    <motion.div
      className="modal-backdrop"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      onClick={onClose}
    >
      <motion.div
        className="modal-content"
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3>Why this paper was recommended</h3>

        <ExplanationSection explanation={explanation} />

        <button className="close-btn" onClick={onClose}>
          Close
        </button>
      </motion.div>
    </motion.div>,
    document.getElementById("modal-root")!
  )
}