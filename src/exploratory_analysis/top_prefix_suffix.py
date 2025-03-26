import pandas as pd
from collections import Counter


def get_top_prefix_suffix(df, category_col="Category", entity_col="Entity", top_n=10):
	results = []
	grouped = df.groupby(category_col)

	for category, group in grouped:
		prefix_counts = Counter()
		suffix_counts = Counter()

		for entity in group[entity_col]:
			entity_lower = str(entity).lower().strip()
			if len(entity_lower) >= 3:
				prefix = entity_lower[:3]
				suffix = entity_lower[-3:]
				prefix_counts[prefix] += 1
				suffix_counts[suffix] += 1

		top_prefixes = prefix_counts.most_common(top_n)
		top_suffixes = suffix_counts.most_common(top_n)

		# store results
		results.append(
			{"Category": category, "Top Prefixes (3 letters)": top_prefixes, "Top Suffixes (3 letters)": top_suffixes}
		)

	return pd.DataFrame(results)


df = pd.read_csv("entities/all_entities.csv")
prefix_suffix_df = get_top_prefix_suffix(df, top_n=10)
prefix_suffix_df.to_csv("prefix_suffix.csv", index=False)
print(prefix_suffix_df)
