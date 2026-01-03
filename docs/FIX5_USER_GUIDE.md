# Fix 5: Small Model Template - User Guide

## What is Fix 5?

Fix 5 provides **optimized prompt templates for small models** (phi3:mini, tinyllama, 1b-3b parameter models). These simplified templates help small models generate accurate code by reducing cognitive load and providing clear patterns.

## When Fix 5 Activates

The system automatically detects your model size and uses the appropriate template:

### Small Models (Simple Template):
- phi3:mini, phi3:mini-4k
- tinyllama, tinyllama:latest
- qwen2:1.5b, qwen2:3b
- gemma:2b
- Any model with "mini" or "tiny" in the name
- Models with 1b-3b parameters

### Large Models (Detailed Template):
- llama3.1:8b and larger
- qwen2:7b and larger
- mixtral:8x7b
- All models 7b+ parameters

## What Fix 5 Does Well

### ✅ Excellent Performance (80-100% accuracy):
- **Basic aggregations**: max, min, sum, mean, count
- **Statistical functions**: median, standard deviation, variance
- **Simple ranking**: top N, bottom N
- **Direct lookups**: "which has highest X"

### ✅ Example Queries That Work Great:
```
"What is the maximum sales?"
"Calculate the average revenue"
"Show me the top 10 products by price"
"What is the standard deviation of revenue?"
"Find the total marketing spend"
"Which product has the highest sales?"
```

## Known Limitations

### ⚠️ Challenging (50-70% accuracy):
- **Complex filtering**: Multiple conditions combined
- **Nested operations**: Multiple steps in one query
- **Custom calculations**: Formulas not in common patterns

### ❌ Problematic (0-30% accuracy):
- **Ambiguous queries** without clear metric
- **Multiple conflicting requirements**
- **Queries requiring external knowledge**

### ❌ Example Queries That May Fail:
```
"Which one is highest?"  
→ Problem: Highest what? Sales, revenue, price?
→ Solution: Be specific - "Which product has the highest sales?"

"Show me the best products"
→ Problem: "Best" by what metric?
→ Solution: "Show me the top 10 products by revenue"

"Find products where sales are above average and in the North region"
→ Problem: Multiple conditions challenging for small models
→ Solution: Use larger model or break into steps
```

## How to Get Best Results

### 1. Be Specific About Metrics
❌ "What's the highest?"  
✅ "What is the highest sales value?"

### 2. Use Clear Actions
❌ "Tell me about top products"  
✅ "Show me the top 5 products by revenue"

### 3. One Operation at a Time
❌ "Find products with sales > 5000 and price < 50 sorted by revenue"  
✅ "Show me products with sales greater than 5000"

### 4. Use Standard Terms
✅ "maximum", "minimum", "average", "total", "top N"  
⚠️ "biggest", "smallest", "best", "worst" (less reliable)

## When to Use a Larger Model

Consider switching to a larger model (llama3.1:8b+) if you need:

1. **Complex multi-condition filtering**
   - "Find products where (sales > 5000 OR revenue > 50000) AND region is NOT 'North'"

2. **Machine learning operations**
   - "Cluster these products into 3 groups"
   - "Predict sales using regression"

3. **Statistical tests**
   - "Perform t-test between North and South regions"
   - "Calculate correlation matrix"

4. **Custom formulas**
   - "Calculate ROI as (revenue - marketing_spend) / marketing_spend * 100"

## Troubleshooting

### "Code execution failed"
**Cause**: Small model generated invalid syntax  
**Solution**: 
1. Rephrase query to be more explicit
2. Try the query again (small models have ~10-20% variance)
3. Use a larger model if query is inherently complex

### "Wrong result returned"
**Cause**: Model misunderstood which metric to use  
**Solution**:
1. Explicitly mention the column name
2. Use standard aggregation terms (max, min, average, sum)

### "No result" or "Empty result"
**Cause**: Model filtered too aggressively or query was ambiguous  
**Solution**:
1. Verify your data has matching rows
2. Simplify the query
3. Check if column names are correct

## Performance Guidelines

### Expected Response Times:
- **Simple queries** (max, min, sum): 3-8 seconds
- **Moderate queries** (top N, filtering): 5-15 seconds
- **Statistical queries** (std dev, median): 8-20 seconds

### Expected Accuracy:
- **Basic aggregations**: ~90% correct
- **Statistical operations**: ~90% correct
- **Filtering/ranking**: ~70% correct
- **Ambiguous queries**: ~10-30% correct

### Tips for Speed:
1. Use small models (phi3:mini) for simple queries - 2x faster than large models
2. Be specific in your query - reduces retry attempts
3. Avoid asking multiple questions at once

## Summary

Fix 5 makes small models **reliable for common data analysis tasks**. For best results:

✅ **Do**: Use clear, specific queries with standard terms  
✅ **Do**: Focus on one operation at a time  
✅ **Do**: Mention column names explicitly  

❌ **Don't**: Use ambiguous language ("best", "which one")  
❌ **Don't**: Combine multiple complex conditions  
❌ **Don't**: Expect small models to handle advanced statistics  

**When in doubt**: Try a larger model (llama3.1:8b) for complex queries.

---

**Questions?** Check the comprehensive test results in `FIX5_TEMPLATE_IMPROVEMENTS.md` for detailed performance data.
