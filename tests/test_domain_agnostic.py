import pytest
from unittest.mock import patch, MagicMock

def test_domain_agnostic_analysis(client, tmp_path):
    """
    Test that the system can handle arbitrary domain data (e.g., Alien metrics)
    and correctly passes this schema to the LLM.
    """
    # 1. Create 'Alien' Data
    alien_csv = tmp_path / "alien_data.csv"
    alien_csv.write_text("Sector,GlargBits,Xylophars\nAlpha,100,5.5\nBeta,200,6.6\nGamma,300,7.7")
    
    # 2. Upload
    with open(alien_csv, "rb") as f:
        files = {"file": ("alien_data.csv", f, "text/csv")}
        up_resp = client.post("/api/upload/", files=files)
    assert up_resp.status_code == 200
    filename = up_resp.json()["filename"]
    
    # 3. Analyze with Mocked LLM - INSPECT PROMPT
    # Force CoT disabled to ensure direct LLM call
    with patch("backend.core.llm_client.LLMClient.generate") as mock_gen, \
         patch("backend.plugins.data_analyst_agent.DataAnalystAgent._load_cot_config", return_value={'enabled': False}):
        
        mock_gen.return_value = {"response": "The average GlargBits is 200."}
        
        payload = {
            "query": "Calculate average GlargBits per sector",
            "filename": filename,
            "session_id": "test_domain_1"
        }
        resp = client.post("/api/analyze/", json=payload)
        
        assert resp.status_code == 200
        data = resp.json()
        print(f"\nDEBUG_DATA: {data}\n")
        assert data["status"] == "success"
        
        # 4. Verify Prompt Content (White Box Accuracy Check)
        calls = mock_gen.call_args_list
        assert len(calls) > 0, f"LLM was not called! Calls: {len(calls)}"
        
        # Access args safely
        last_call = calls[0]
        # unittest.mock.call is a tuple (args, kwargs)
        args_tuple = last_call[0]
        kwargs_dict = last_call[1]
        
        prompt = ""
        if len(args_tuple) > 0:
            prompt = args_tuple[0]
        elif "prompt" in kwargs_dict:
            prompt = kwargs_dict["prompt"]
        else:
            # Fallback for debugging - maybe it was called with a different first arg?
            prompt = str(args_tuple) + str(kwargs_dict)
            
        print(f"\nDEBUG_PROMPT_PREVIEW: {str(prompt)[:50]}...\n")
        
        assert "GlargBits" in prompt
        assert "Xylophars" in prompt
        assert "Sector" in prompt
        
        print(f"\nDEBUG_PROMPT:\n{prompt}\n")
        
        assert "GlargBits" in prompt
        assert "Xylophars" in prompt
        assert "Sector" in prompt
        # Verify stats are present (e.g. 100, 300)
        assert "100" in prompt
        assert "300" in prompt
