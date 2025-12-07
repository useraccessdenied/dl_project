import json
import csv

def json_to_csv(json_file_path, csv_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Define CSV headers
        headers = [
            'Seed Question', 
            'Level', 
            'Follow-up Question',  # Included for context
            'Logical Consistency', 
            'Informativeness', 
            'Relevance', 
            'Clarity', 
            'Aggregate Score',
            'ChatGPT Logical Consistency', 
            'ChatGPT Informativeness', 
            'ChatGPT Relevance', 
            'ChatGPT Clarity'
        ]

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for entry in data:
                seed_question = entry.get('seed_question', '')
                evaluations = entry.get('evaluation', {})

                # Iterate through each level (level_2, level_3, etc.)
                for level_key, stats in evaluations.items():
                    # Main Model Scores
                    question_text = stats.get('question', '')
                    log_con = stats.get('logical_consistency', '')
                    info = stats.get('informativeness', '')
                    rel = stats.get('relevance', '')
                    clarity = stats.get('clarity', '')
                    agg_score = stats.get('aggregate_score', '')

                    # ChatGPT Scores
                    chatgpt_stats = stats.get('chatgpt', {})
                    gpt_log_con = chatgpt_stats.get('logical_consistency', '')
                    gpt_info = chatgpt_stats.get('informativeness', '')
                    gpt_rel = chatgpt_stats.get('relevance', '')
                    gpt_clarity = chatgpt_stats.get('clarity', '')

                    row = [
                        seed_question,
                        level_key,
                        question_text,
                        log_con,
                        info,
                        rel,
                        clarity,
                        agg_score,
                        gpt_log_con,
                        gpt_info,
                        gpt_rel,
                        gpt_clarity
                    ]
                    
                    writer.writerow(row)

        print(f"Successfully converted '{json_file_path}' to '{csv_file_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

json_to_csv('bfqg_recursive_results.json', 'bfqg_results.csv')