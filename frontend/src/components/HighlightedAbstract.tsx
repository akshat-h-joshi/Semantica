interface Props {
  abstract: string
  keywords: string[]
}

export default function HighlightedAbstract({ abstract, keywords }: Props) {
  if (!keywords || keywords.length === 0) {
    return <p className="abstract">{abstract}</p>
  }

  const escaped = keywords.map(k =>
    k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  )

  const regex = new RegExp(`(${escaped.join("|")})`, "gi")

  const parts = abstract.split(regex)

  return (
    <p className="abstract">
      {parts.map((part, i) =>
        keywords.some(k => k.toLowerCase() === part.toLowerCase()) ? (
          <mark key={i} className="keyword-highlight">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </p>
  )
}