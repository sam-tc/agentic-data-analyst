import argparse

from src.agent import ask_agent
from src.data_loader import load_csv


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about CSV datasets."
    )

    parser.add_argument("dataset", help="Path to a CSV file")
    parser.add_argument("question", help="Natural-language question")

    args = parser.parse_args()

    df = load_csv(args.dataset)

    answer, trace = ask_agent(df, args.question)

    print("\nANSWER")
    print(answer)

    print("\nACTUAL TOOL EXECUTION")

    for event in trace:
        print(
            f"- {event['tool']}: {event['arguments']}"
        )

        if "error" in event["result"]:
            print(f"  Error: {event['result']['error']}")

        if "chart_file" in event["result"]:
            print(
                f"  Chart: {event['result']['chart_file']}"
            )


if __name__ == "__main__":
    main()
