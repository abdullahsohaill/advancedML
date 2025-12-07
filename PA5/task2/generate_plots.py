import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- SETUP ---
# Set a professional scientific style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
PALETTE = "viridis" 

def load_data():
    try:
        df_eval = pd.read_csv("final_evaluation_results.csv")
        try:
            df_pert = pd.read_csv("rm_perturbation_results.csv")
        except:
            df_pert = None
            print("Warning: rm_perturbation_results.csv not found. Figure 6 will be skipped.")
            
        return df_eval, df_pert
    except FileNotFoundError:
        print("Error: final_evaluation_results.csv not found. Did you run the evaluation script?")
        return None, None

# --- FIGURE 3: VERBOSITY BIAS (Box Plot) ---
def plot_verbosity_bias(df):
    plt.figure(figsize=(10, 6))
    
    # Filter for Open-Ended questions only
    subset = df[df["Category"] == "Verbosity_Probe"]
    
    # Create Box Plot
    sns.boxplot(x="Model", y="Length", data=subset, palette=PALETTE)
    
    # REMOVED TITLE
    # plt.title("Figure 3: Verbosity Bias (Response Length Distribution)", fontweight="bold")
    
    plt.ylabel("Token Count")
    plt.xlabel("")
    
    # Annotation
    plt.figtext(0.5, 0.01, "Higher median and longer whiskers indicate 'rambling' behavior.", 
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig("Fig3_Verbosity.png", dpi=300)
    print("✅ Saved Figure 3: Verbosity Bias")

# --- FIGURE 4: SAFETY vs DRIFT (Scatter Plot) ---
def plot_safety_tradeoff(df):
    plt.figure(figsize=(9, 7))
    
    # Filter for Safety Hacks and calculate MEAN per model
    subset = df[df["Category"] == "Hack_Safety"].groupby("Model")[["KL_Div", "Reward"]].mean().reset_index()
    
    # Create Scatter Plot with large points
    sns.scatterplot(data=subset, x="KL_Div", y="Reward", hue="Model", style="Model", s=400, palette=PALETTE)
    
    # Draw Reference Lines
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Annotations for quadrants
    plt.text(subset["KL_Div"].max(), subset["Reward"].max(), "High Reward / High Drift\n(Aggressive Alignment)", 
             ha='right', va='bottom', color='darkred', fontsize=10, weight='bold')
    
    # REMOVED TITLE
    # plt.title("Figure 4: The Alignment Trade-off (Safety Reward vs. KL Drift)", fontweight="bold")
    
    plt.xlabel("KL Divergence (Drift from Base Model)")
    plt.ylabel("Safety Reward (Higher = Better Refusal)")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Fig4_Safety_Tradeoff.png", dpi=300)
    print("✅ Saved Figure 4: Safety Trade-off")

# --- FIGURE 5: CONSTRAINT COMPLIANCE (Dual Bar Chart) ---
def plot_constraints(df):
    # Filter for Constraints
    subset = df[df["Category"] == "Hack_Constraint"].copy()
    
    # Calculate Stats
    stats = subset.groupby("Model").agg({
        "Compliant": "mean",  # Compliance Rate (0-1)
        "Deviation": "mean"   # Avg words over limit
    }).reset_index()
    
    # Setup Dual Plot
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot A: Compliance Rate
    ax0 = plt.subplot(gs[0])
    sns.barplot(data=stats, x="Model", y="Compliant", palette=PALETTE, ax=ax0)
    ax0.set_title("A. Compliance Rate (%)", fontweight="bold")
    ax0.set_ylim(0, 1.1)
    ax0.set_ylabel("Rate (1.0 = Perfect)")
    ax0.set_xlabel("")
    # Add percentage text
    for i, v in enumerate(stats["Compliant"]):
        ax0.text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontweight='bold')

    # Plot B: Deviation Magnitude
    ax1 = plt.subplot(gs[1])
    sns.barplot(data=stats, x="Model", y="Deviation", palette="magma", ax=ax1)
    ax1.set_title("B. Magnitude of Failure (Words Over Limit)", fontweight="bold")
    ax1.set_ylabel("Avg Words Over Limit")
    ax1.set_xlabel("")
    
    # REMOVED MAIN FIGURE TITLE
    # plt.suptitle("Figure 5: Instruction Following Performance", fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.savefig("Fig5_Constraints.png", dpi=300)
    print("✅ Saved Figure 5: Constraints")

# --- FIGURE 6: RM SENSITIVITY (Perturbation Test) ---
def plot_rm_sensitivity(df_pert):
    if df_pert is None: return

    plt.figure(figsize=(10, 6))
    
    # Calculate the shift: New Score - Base Score
    base_score = df_pert[df_pert["Type"] == "Base"]["Reward"].values[0]
    df_pert["Reward Shift"] = df_pert["Reward"] - base_score
    
    # Remove the "Base" row so we only see changes
    plot_data = df_pert[df_pert["Type"] != "Base"]
    
    # Bar Chart
    sns.barplot(data=plot_data, x="Type", y="Reward Shift", palette="coolwarm")
    plt.axhline(0, color='black', linewidth=1.5)
    
    # REMOVED TITLE
    # plt.title("Figure 6: Reward Model Overparameterization Test", fontweight="bold")
    
    plt.ylabel("Reward Shift vs. Base Prompt")
    plt.xlabel("Perturbation Type")
    
    # Annotation
    plt.figtext(0.5, 0.01, "Bars indicate how much the score changes just by adding superficial 'fluff'.", 
                ha="center", fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig("Fig6_RM_Sensitivity.png", dpi=300)
    print("✅ Saved Figure 6: RM Sensitivity")

# --- EXECUTION ---
if __name__ == "__main__":
    df, df_pert = load_data()
    
    if df is not None:
        plot_verbosity_bias(df)
        plot_safety_tradeoff(df)
        plot_constraints(df)
        plot_rm_sensitivity(df_pert)
        print("\nAll plots generated successfully!")