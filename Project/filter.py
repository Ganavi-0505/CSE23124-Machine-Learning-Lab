import pandas as pd
from datetime import datetime, timedelta

start_date = datetime.strptime('2020-07-01', '%Y-%m-%d')
end_date = datetime.strptime('2020-09-30', '%Y-%m-%d')

failures_list = []

current_date = start_date
while current_date <= end_date:
    filename = current_date.strftime('%Y-%m-%d') + '.csv'
    try:
        print(f"Processing {filename}...")
        df = pd.read_csv(filename, low_memory=False)  # Reduce risk of read errors

        if 'failure' in df.columns:
            filtered_df = df[df['failure'] > 0].copy()  # Avoid SettingWithCopyWarning
            if not filtered_df.empty:
                filtered_df.loc[:, 'source_file'] = filename  # Safe assignment
                failures_list.append(filtered_df)

    except FileNotFoundError:
        print(f"Warning: {filename} not found. Skipping...")
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    current_date += timedelta(days=1)

# Save results
if failures_list:
    all_failures = pd.concat(failures_list, ignore_index=True)
    all_failures.to_csv('failures.csv', index=False)
    print("Filtered failures saved to failures.csv.")
else:
    print("No failures found in the specified date range.")
