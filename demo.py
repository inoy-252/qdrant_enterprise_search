import time

from qdrant_manager import QdrantSemanticCache


def main():
    print("=" * 60)
    print("ENTERPRISE QDRANT SEMANTIC CACHE - INTERACTIVE TEST")
    print("=" * 60)

    cache = QdrantSemanticCache(threshold=0.85)

    while True:
        print("\n--- MENU ---")
        print("1. Store a new Query-Response pair")
        print("2. Search Cache (Test Semantic Lookup)")
        print("3. View Live Qdrant Telemetry (get_stats)")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            q = input("Enter Question / Query: ").strip()
            r = input("Enter Answer / Response: ").strip()
            dept = input("Enter Department (optional, press Enter to skip): ").strip()

            meta = {"department": dept} if dept else None
            point_id = cache.store(q, r, metadata=meta)
            print(f"SUCCESS: Stored in Qdrant! Point ID: {point_id}")

        elif choice == "2":
            query = input("Enter query to search: ").strip()
            dept_filter = input(
                "Filter by Department (optional, press Enter to skip): "
            ).strip()

            filter_meta = {"department": dept_filter} if dept_filter else None

            start_time = time.time()
            hit = cache.lookup(query, filter_metadata=filter_meta)
            latency = (time.time() - start_time) * 1000  # in milliseconds

            print(f"\nSearch completed in {latency:.2f}ms:")
            if hit:
                print(f"CACHE HIT! (Similarity: {hit['similarity'] * 100:.1f}%)")
                print(f"  Matched Query: {hit['query']}")
                print(f"  Response:      {hit['response']}")
                print(f"  Metadata:      {hit['metadata']}")
            else:
                print("CACHE MISS (No match found above threshold)")

        elif choice == "3":
            stats = cache.get_stats()
            print("\nLIVE QDRANT TELEMETRY:")
            for key, val in stats.items():
                print(f"  {key}: {val}")

        elif choice == "4":
            print("Exiting demo. Great work!")
            break


if __name__ == "__main__":
    main()
