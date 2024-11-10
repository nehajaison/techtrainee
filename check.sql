-- Step 1: Create the Database
CREATE DATABASE user_database;

-- Step 2: Use the Database
USE user_database;

-- Step 3: Create the User Table
CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_name VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL
);

-- Step 4: Insert a Test User (use a plain password for testing purposes)
INSERT INTO user (user_name, password, name) VALUES
('testuser', 'testpassword', 'Test User');
