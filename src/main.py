import argparse
from src.eval.evaluation import evaluate
from src.models.recommender import run_recommendation
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Loading papers...")

def main(mode, model):
    if mode == "recommend":
        query = input("What subject would you like to explore? ")
        recommendations = run_recommendation(query, model)

        logger.info("\nTop recommendations:\n")
        for title, score in recommendations:
            logger.info(f"{score:.3f} — {title}")

    elif mode == "evaluate":
        metrics = evaluate()

        logger.info("Evaluation Results:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name.upper()}")
            for metric, value in model_metrics.items():
                logger.info(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="recommend",
        choices=["recommend","evaluate"],
        help="Run mode"
    )
    parser.add_argument(
        "--model",
        default="mpnet",
        choices=["mini", "mpnet", "tfidf", "hybrid"],

    )
    args = parser.parse_args()
    main(mode=args.mode, model=args.model)



