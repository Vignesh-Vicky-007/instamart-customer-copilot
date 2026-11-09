-- Part A SQL analysis queries (run against data/processed/instamart.db)

-- Revenue and orders by city (orders JOIN customers)
SELECT c.city, COUNT(*) AS orders, ROUND(SUM(o.order_amount),2) AS revenue FROM orders o JOIN customers c ON c.customer_id=o.customer_id WHERE o.status='completed' GROUP BY c.city ORDER BY revenue DESC;

-- Top 5 products by revenue (orders JOIN products)
SELECT p.product_name, p.category, SUM(o.quantity) AS units, ROUND(SUM(o.order_amount),2) AS revenue FROM orders o JOIN products p ON p.product_id=o.product_id WHERE o.status='completed' GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5;

-- Average delivery minutes by product category (3-table join)
SELECT p.category, ROUND(AVG(o.delivery_minutes),1) AS avg_delivery_min, COUNT(*) AS orders FROM orders o JOIN products p ON p.product_id=o.product_id JOIN customers c ON c.customer_id=o.customer_id WHERE o.status='completed' GROUP BY p.category ORDER BY avg_delivery_min DESC;

-- Customers with the most tickets (tickets JOIN customers)
SELECT t.customer_id, c.city, COUNT(*) AS tickets FROM tickets t JOIN customers c ON c.customer_id=t.customer_id GROUP BY t.customer_id ORDER BY tickets DESC, t.customer_id LIMIT 5;

-- Orders whose customer is missing (integrity check, expect 0)
SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON c.customer_id=o.customer_id WHERE c.customer_id IS NULL;

