import pandas as pd

# Load the combined RMS dataset
df = pd.read_csv("uae_telecom_rms_dataset.csv")

# Define sector mappings: {sector value in CSV: output filename}
sectors = {
    "Network Infrastructure": "network_infrastructure_rms_dataset.csv",
    "IT & Software Development": "it_software_development_rms_dataset.csv",
    "Telecom Services": "telecom_services_rms_dataset.csv",
    "Customer Support & Delivery": "customer_support_delivery_rms_dataset.csv",
    "Regulatory Compliance": "regulatory_compliance_rms_dataset.csv",
    "Supply Chain Management": "supply_chain_management_rms_dataset.csv",
    "Cybersecurity": "cybersecurity_rms_dataset.csv",
    "Marketing & Sales": "marketing_sales_rms_dataset.csv",
    "Human Resources": "human_resources_rms_dataset.csv",
    "Research & Development": "research_development_rms_dataset.csv"
}

for sector, filename in sectors.items():
    sector_df = df[df['sector'] == sector]
    sector_df.to_csv(filename, index=False)
