import "../App.css"

interface Props {
  model: string
  setModel: (model: string) => void
}

export default function ModelSelector({ model, setModel }: Props) {
  return (
    <select className="model-selector"
      value={model}
      onChange={(e) => setModel(e.target.value)}
    >
      <option value="mpnet">MPNet (semantic)</option>
      <option value="mini">MiniLM (semantic)</option>
      <option value="tfidf">TF-IDF (lexical)</option>
      <option value="hybrid">Hybrid (MPNet + TF-IDF)</option>
    </select>
  )
}