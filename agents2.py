import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure LLM
llm = ChatGroq(
    model='mixtral-8x7b-32768', 
    temperature=0.3, 
    api_key=os.getenv('GROQ_API_KEY')
)

# Initialize Serper Tool for real-time data fetching
serper_tool = SerperDevTool()

# Agents
def create_research_agent():
    return Agent(
        role="Investment Report Researcher",
        goal="Generate comprehensive and accurate investment reports",
        backstory=(
            "An expert financial analyst with deep understanding of "
            "investment performance metrics and market trends."
        ),
        llm=llm,
        tools=[serper_tool],
        allow_delegation=True
    )

def create_compliance_agent():
    return Agent(
        role="Regulatory Compliance Specialist",
        goal="Ensure investment reports meet industry regulations",
        backstory=(
            "A meticulous professional specializing in financial "
            "reporting regulations and compliance standards."
        ),
        llm=llm,
        tools=[serper_tool],
        allow_delegation=False
    )

def create_writer_agent():
    return Agent(
        role="Investment Report Writer", 
        goal="Craft clear, insightful, and client-specific investment reports", 
        backstory=(
            "A skilled communicator translating complex financial "
            "data into understandable insights for diverse clients."
        ),
        llm=llm,
        tools=[serper_tool],
        allow_delegation=False
    )

# Tasks
def create_research_task(agent, client_profile, portfolio_data):
    return Task(
        description=(
            f"Analyze portfolio performance data for {client_profile} client. "
            f"Conduct real-time market research using Serper. "
            f"Key inputs: Risk Tolerance={portfolio_data['risk_tolerance']}, "
            f"Investment Horizon={portfolio_data['investment_horizon']}, "
            f"Total Return={portfolio_data['total_return']}%, "
            f"Risk Metric={portfolio_data['risk_metric']}%, "
            f"Asset Allocation={', '.join(portfolio_data['asset_allocation'])}. "
            "Extract key performance metrics, market trends, and prepare "
            "comprehensive insights for the report."
        ),
        expected_output=(
            "Detailed analysis including:"
            "- Performance metrics"
            "- Asset allocation insights"
            "- Market trend research"
            "- Comparative benchmarks"
        ),
        agent=agent,
        context=[
            {
                "description": "Portfolio Data Context",
                "expected_output": "Provide context for investment research",
                "data": portfolio_data
            }
        ]
    )
def create_compliance_task(agent, client_profile):
    return Task(
        description=(
            f"Review investment report for {client_profile} client. "
            "Verify compliance with industry regulations, ensure "
            "accuracy of disclosures, and validate report content."
        ),
        expected_output=(
            "Compliance verification report including:"
            "- Regulatory checklist"
            "- Disclosure review"
            "- Recommended amendments"
        ),
        agent=agent
    )

def create_writing_task(agent, client_profile):
    return Task(
        description=(
            f"Generate a comprehensive investment report for {client_profile} client. "
            "Synthesize research findings and compliance checks into "
            "a clear, personalized report."
        ),
        expected_output=(
            "Final investment report including:"
            "- Executive summary"
            "- Detailed performance analysis"
            "- Personalized insights"
            "- Forward-looking recommendations"
        ),
        agent=agent,
        output_file=f'{client_profile}_investment_report.md'
    )

# Streamlit App
def main():
    st.title("AI Investment Report Generator")
    
    # Sidebar for client profile selection
    st.sidebar.header("Client Profile Configuration")
    
    # Client profile inputs
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance", 
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    investment_horizon = st.sidebar.selectbox(
        "Investment Horizon", 
        ["Short-term", "Medium-term", "Long-term"]
    )
    
    # Portfolio performance inputs
    st.header("Portfolio Performance Data")
    
    # Performance metric inputs
    total_return = st.number_input("Total Portfolio Return (%)", min_value=0.0, max_value=100.0, step=0.1, value=15.6)
    risk_metric = st.number_input("Portfolio Risk Metric (Volatility %)", min_value=0.0, max_value=100.0, step=0.1, value=22.3)
    
    # Asset allocation input
    asset_allocation = st.text_area(
        "Asset Allocation (comma-separated)", 
        value="US Stocks, International Stocks, Emerging Markets, Bonds, Real Estate"
    )
    
    # Generate Report Button
    if st.button("Generate Investment Report"):
        # Prepare portfolio data
        portfolio_data = {
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon,
            "total_return": total_return,
            "risk_metric": risk_metric,
            "asset_allocation": [alloc.strip() for alloc in asset_allocation.split(',')]
        }
        
        # Create agents
        researcher = create_research_agent()
        compliance_agent = create_compliance_agent()
        writer = create_writer_agent()
        
        # Create tasks
        client_profile = f"{risk_tolerance}_{investment_horizon}"
        research_task = create_research_task(researcher, client_profile, portfolio_data)
        compliance_task = create_compliance_task(compliance_agent, client_profile)
        writing_task = create_writing_task(writer, client_profile)
        
        # Create crew
        crew = Crew(
            agents=[researcher, compliance_agent, writer],
            tasks=[research_task, compliance_task, writing_task],
            process=Process.sequential
        )
        
        # Generate report
        with st.spinner('Generating Investment Report...'):
            outcome = crew.kickoff(inputs=portfolio_data)
        
        # Display results
        st.success("Investment Report Generated Successfully!")
        st.download_button(
            label="Download Report",
            data=outcome,
            file_name=f'{client_profile}_investment_report.md',
            mime='text/markdown'
        )
        
        # Show report preview
        st.subheader("Report Preview")
        st.code(outcome, language='markdown')

if __name__ == "__main__":
    main()