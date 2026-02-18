# Test Instructions for Automation Site

## Overview
These test instructions are designed to be **challenging** and **learning-oriented**. Use them to test the site's automation capabilities and identify areas for improvement.

---

## Level 1: Basic Navigation & Login Tests

### Test 1.1: Simple Login
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login with the standard user account
```
**Expected:** Should successfully login and navigate to products page.
**Learning:** Tests basic form filling and navigation.

---

### Test 1.2: Login with Invalid Credentials
**Website:** https://www.saucedemo.com  
**Username:** invalid_user  
**Password:** wrong_password  
**Instruction:**
```
Try to login and check for error message
```
**Expected:** Should display error and remain on login page.
**Learning:** Tests error handling and message detection.

---

## Level 2: Multi-Step Interactions

### Test 2.1: Complete Purchase Flow
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, add the sauce labs backpack to cart, proceed to checkout, and verify the item appears in checkout
```
**Expected:** Should navigate through login → products → add to cart → checkout with item visible.
**Learning:** Tests complex multi-step workflows.

---

### Test 2.2: Filtering & Selection
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, sort products by price low to high, and add the cheapest item to cart
```
**Expected:** Should login, apply filter, identify lowest price item, and add it.
**Learning:** Tests dynamic content handling and sorting logic.

---

## Level 3: Advanced Form Handling

### Test 3.1: Complex Form Filling
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, add any item to cart, go to checkout, and fill in first name "John", last name "Doe", and postal code "12345"
```
**Expected:** Should complete all fields and show order summary.
**Learning:** Tests form field identification and multi-field filling.

---

### Test 3.2: Conditional Navigation
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, check if there are more than 3 products available, and if yes, add the third product to cart
```
**Expected:** Should evaluate condition and add product if criteria met.
**Learning:** Tests conditional logic and element counting.

---

## Level 4: Edge Cases & Robustness

### Test 4.1: Handling Dynamic Content
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, wait for all products to load completely, then add the first product and verify cart count updates
```
**Expected:** Should handle loading delays and verify cart update.
**Learning:** Tests robustness with dynamic/slow-loading content.

---

### Test 4.2: Multiple Item Management
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, add three different items to cart, then remove the middle one and verify only two items remain
```
**Expected:** Should manage cart state correctly after additions and removals.
**Learning:** Tests state management across multiple interactions.

---

## Level 5: Text Extraction & Verification

### Test 5.1: Content Verification
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login and verify that the page title contains the word "Swag"
```
**Expected:** Should detect and verify page title content.
**Learning:** Tests text extraction and validation.

---

### Test 5.2: Price Comparison
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, find the product with price closest to $15, add it to cart, and that its price appears in checkout
```
**Expected:** Should identify pricing, add product, and verify in different context.
**Learning:** Tests numerical comparison and cross-context verification.

---

## Level 6: Real-World Scenarios

### Test 6.1: User Profile Update
**Website:** https://www.saucedemo.com  
**Username:** problem_user  
**Password:** secret_sauce  
**Instruction:**
```
Login with the problem_user account and attempt to add an item to the cart
```
**Expected:** Should either succeed or clearly report the issue.
**Learning:** Tests handling of different user types and potential bugs.

---

### Test 6.2: Logout & Re-login
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, add an item to cart, logout by clicking the menu button and selecting logout, then login again and check if cart is empty
```
**Expected:** Should handle logout and verify cart reset.
**Learning:** Tests session handling and state persistence.

---

## Level 7: Stress Testing

### Test 7.1: Rapid Interactions
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login and quickly add four products to cart without waiting between clicks
```
**Expected:** Should handle rapid interactions without errors.
**Learning:** Tests performance and race condition handling.

---

### Test 7.2: Long Form Interaction
**Website:** https://www.saucedemo.com  
**Username:** standard_user  
**Password:** secret_sauce  
**Instruction:**
```
Login, add an item, go to checkout, fill all required fields, and complete the entire purchase flow
```
**Expected:** Should complete full workflow and show confirmation.
**Learning:** Tests end-to-end workflow reliability.

---

## Scoring & Metrics

### Success Criteria
- ✅ Completes within expected steps
- ✅ Extracts correct information
- ✅ Handles errors gracefully
- ✅ Generates clean, working code
- ✅ Provides accurate screenshots

### Areas to Improve If Tests Fail
1. **Form Handling:** Element identification, value injection
2. **Navigation:** URL changes, redirect handling
3. **Waits:** Dynamic content loading, element appearance
4. **Error Detection:** Error message parsing, retry logic
5. **State Management:** Cart consistency, session handling
6. **Text Extraction:** Price parsing, content verification

---

## Tips for Testing

1. **Start Simple:** Begin with Level 1 before advancing
2. **Observe Failures:** Note exactly what failed and why
3. **Vary Websites:** Try different sites beyond Sauce Demo
4. **Check Logs:** Review execution steps and generated code
5. **Iterate:** Refine instructions based on results

---

## Alternative Test Websites

- **E-commerce:** https://demo.nopcommerce.com/
- **Forms:** https://formy.herokuapp.com/
- **Interactive:** https://the-internet.herokuapp.com/
- **Dynamic:** https://automatetheplanet.com/multiple-elements-multiplicity/

