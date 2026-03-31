CREATE DATABASE Bank_Customer_Churn;
USE Bank_Customer_Churn;

CREATE TABLE Customer (
	CustomerID INT PRIMARY KEY,
    CreditScore INT,
    Age INT,
    Gender VARCHAR(20),
    Geography VARCHAR(50)
);

CREATE TABLE Account (
	AccountID INT PRIMARY KEY,
    CustomerID INT,
    Balance DECIMAL(12, 2),
    NumberOFProducts INT,
    CreditCardOwnership BOOLEAN,
    ActiveMembership BOOLEAN,
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

CREATE TABLE ChurnStatus (
	CustomerID INT PRIMARY KEY,
    ChurnIndicator BOOLEAN,
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

INSERT INTO customer (CustomerID, CreditScore, Age, Gender, Geography)
VALUES (1, 720, 34, 'Male', 'USA'),
(2, 650, 45, 'Female', 'France'),
(3, 580, 29, 'Male', 'Spain'),
(4, 810, 52, 'Female', 'Germany'),
(5, 690, 41, 'Male', 'USA'),
(6, 730, 37, 'Female', 'Italy'),
(7, 600, 28, 'Male', 'Canada'),
(8, 770, 50, 'Female', 'USA'),
(9, 640, 33, 'Male', 'Spain'),
(10, 710, 39, 'Female', 'France');

INSERT INTO Account (AccountID, CustomerID, Balance, NumberOfProducts, CreditCardOwnership, ActiveMembership)
VALUES
(101, 1, 1500.75, 2, TRUE, TRUE),
(102, 2, 3200.10, 1, FALSE, TRUE),
(103, 3, 500.00, 1, TRUE, FALSE),
(104, 4, 8900.55, 3, TRUE, TRUE),
(105, 5, 250.20, 1, FALSE, FALSE),
(106, 6, 4100.00, 2, TRUE, TRUE),
(107, 7, 1200.90, 1, FALSE, TRUE),
(108, 8, 7600.40, 2, TRUE, TRUE),
(109, 9, 980.00, 1, TRUE, FALSE),
(110, 10, 5400.25, 2, FALSE, TRUE);


INSERT INTO ChurnStatus (CustomerID, ChurnIndicator)
VALUES
(1, FALSE),
(2, TRUE),
(3, FALSE),
(4, FALSE),
(5, TRUE),
(6, FALSE),
(7, TRUE),
(8, FALSE),
(9, FALSE),
(10, TRUE);





	