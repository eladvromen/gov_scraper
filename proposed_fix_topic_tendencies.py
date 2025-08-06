def calculate_topic_tendencies_field_weighted(normalized_data: Dict) -> Dict:
    """
    Calculate topic tendencies with proper field weighting
    Each field gets equal weight regardless of number of values
    """
    logger.info("Calculating field-weighted topic tendencies")
    
    topic_tendencies = {}
    
    for topic in normalized_data:
        model_field_scores = defaultdict(lambda: defaultdict(list))
        
        # First, collect scores by field
        for field in normalized_data[topic]:
            for field_value in normalized_data[topic][field]:
                for model in normalized_data[topic][field][field_value]:
                    data = normalized_data[topic][field][field_value][model]
                    model_field_scores[model][field].append(data['normalized_score'])
        
        # Then, average within each field first, then across fields
        for model in model_field_scores:
            field_averages = []
            for field in model_field_scores[model]:
                if model_field_scores[model][field]:
                    # Average within field
                    field_avg = np.mean(model_field_scores[model][field])
                    field_averages.append(field_avg)
            
            # Average across fields (each field gets equal weight)
            if field_averages:
                topic_tendencies[f"{topic}_{model}"] = np.mean(field_averages)
    
    logger.info("Field-weighted topic tendencies calculated")
    return topic_tendencies

def calculate_topic_tendencies_alternative_methods(normalized_data: Dict) -> Dict:
    """
    Alternative weighting schemes for comparison
    """
    
    # Method 1: Equal field weighting (as above)
    # Method 2: Sample size weighting
    # Method 3: Field count normalization
    
    results = {}
    
    for topic in normalized_data:
        model_scores = defaultdict(list)
        model_weighted_scores = defaultdict(list)
        model_sample_sizes = defaultdict(list)
        
        # Collect all data
        for field in normalized_data[topic]:
            field_scores = defaultdict(list)
            field_weights = defaultdict(list)
            
            for field_value in normalized_data[topic][field]:
                for model in normalized_data[topic][field][field_value]:
                    data = normalized_data[topic][field][field_value][model]
                    
                    # Original method (current)
                    model_scores[model].append(data['normalized_score'])
                    
                    # Field-level data
                    field_scores[model].append(data['normalized_score'])
                    field_weights[model].append(data['total'])  # sample size
            
            # Method 2: Weight by sample size within field
            for model in field_scores:
                if field_scores[model]:
                    # Weighted average within field
                    weights = np.array(field_weights[model])
                    scores = np.array(field_scores[model])
                    field_weighted_avg = np.average(scores, weights=weights)
                    model_weighted_scores[model].append(field_weighted_avg)
        
        # Calculate final scores
        for model in model_scores:
            # Original method
            if model_scores[model]:
                results[f"{topic}_{model}_original"] = np.mean(model_scores[model])
            
            # Field-weighted method
            if model_weighted_scores[model]:
                results[f"{topic}_{model}_field_weighted"] = np.mean(model_weighted_scores[model])
            
            # Field count normalized (divide by number of fields)
            if model_scores[model]:
                num_fields = len(set(field for field in normalized_data[topic]))
                results[f"{topic}_{model}_field_normalized"] = np.mean(model_scores[model]) / num_fields
    
    return results