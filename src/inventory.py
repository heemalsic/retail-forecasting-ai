def simulate_inventory(predictions, initial_stock=50000):
    stock_levels = []
    current_stock = initial_stock

    for demand in predictions:
        current_stock -= demand

        if current_stock < 10000:
            current_stock += 40000  # restock

        stock_levels.append(current_stock)

    return stock_levels