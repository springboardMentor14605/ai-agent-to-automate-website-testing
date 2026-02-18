# Test Examples - Website Reference Guide

## Overview
The test suite has been updated with **diverse websites and different testing scenarios**. No longer limited to just saucedemo.com!

---

## Websites Used

### 1. **Formy.herokuapp.com** (Form Testing)
- **Type:** Dedicated form testing practice site
- **URL:** https://formy.herokuapp.com/form
- **What it tests:**
  - Text input fields
  - Email fields
  - Phone input
  - Date picker
  - Checkbox handling
  - Radio button selection
  - Textarea
  - Form submission
- **Use Cases:** Form filling, validation, multi-step forms

### 2. **Sauce Demo (saucedemo.com)** (E-Commerce)
- **Type:** E-commerce practice site
- **URL:** https://www.saucedemo.com
- **Credentials:**
  - Username: `standard_user`
  - Password: `secret_sauce`
  - Alt Users: `performance_glitch_user`, `problem_user`
- **What it tests:**
  - Login authentication
  - Product browsing
  - Shopping cart operations
  - Checkout flow
  - Filtering/sorting
  - State management
- **Use Cases:** E-commerce workflows, authentication

### 3. **NopCommerce Demo** (E-Commerce)
- **Type:** Real e-commerce platform demo
- **URL:** https://demo.nopcommerce.com/
- **Credentials:** No login required (guest shopping)
- **What it tests:**
  - Product search
  - Category navigation
  - Product filtering
  - Shopping cart
  - Checkout process
  - Multiple product types (electronics, apparel, etc.)
- **Use Cases:** Real-world e-commerce testing

### 4. **The Internet (herokuapp.com/dropdown)** (UI Elements)
- **Type:** UI element practice site
- **URL:** https://the-internet.herokuapp.com/dropdown
- **What it tests:**
  - Dropdown/Select elements
  - JavaScript interactions
  - DOM manipulation
- **Use Cases:** Dropdown handling, dynamic UI elements

### 5. **The Internet (herokuapp.com/windows)** (Multi-Window)
- **Type:** UI element practice site
- **URL:** https://the-internet.herokuapp.com/windows
- **What it tests:**
  - Opening new windows
  - Window switching
  - Multiple browser contexts
- **Use Cases:** Window/tab management

---

## Test Suite Tests (6 Quick Tests)

| # | Test Name | Website | Type | Difficulty |
|---|-----------|---------|------|------------|
| 1 | Form Filling - Text Inputs | Formy | Form | Easy |
| 2 | Sauce Demo - Login & Browse | Sauce Demo | E-Commerce | Easy |
| 3 | NopCommerce - Search Product | NopCommerce | E-Commerce | Medium |
| 4 | Dropdown Selection | The Internet | UI Element | Easy |
| 5 | Checkbox Interaction | Formy | Form | Easy |
| 6 | Radio Button Selection | Formy | Form | Easy |

---

## Testing Panel Scenarios (10 Detailed Tests)

### E-Commerce Category (4 tests)
1. **Complete Purchase Flow** (Sauce Demo)
   - Complexity: Advanced
   - Multi-item cart, checkout flow

2. **NopCommerce - Complete Shopping** (NopCommerce)
   - Complexity: Hard
   - Navigate categories, view specs, add to cart

3. **Search & Add to Cart** (NopCommerce)
   - Complexity: Medium
   - Search, filter results, add quantity

4. **Product Filter & Compare** (NopCommerce)
   - Complexity: Advanced
   - Browse, filter, multi-select, view cart

### Form Handling Category (3 tests)
1. **Multi-Step Form Submission** (Formy)
   - Complexity: Hard
   - Fill all fields: name, email, phone, date, textarea

2. **Checkbox Toggle Test** (Formy)
   - Complexity: Medium
   - Check checkbox element

3. **Date and Time Input** (Formy)
   - Complexity: Medium
   - Date picker, complete form

### UI Elements Category (2 tests)
1. **Handle Multiple Dropdowns** (The Internet)
   - Complexity: Medium
   - Select from dropdown

2. **Radio Button Selection** (The Internet - Formy)
   - Complexity: Easy
   - Select radio option

### Navigation Category (1 test)
1. **Multi-Window Handling** (The Internet)
   - Complexity: Hard
   - Open and verify new windows

---

## Test Coverage by Type

### By Website
- **Formy.herokuapp.com** → 4 tests (form handling)
- **Sauce Demo** → 2 tests (e-commerce + auth)
- **NopCommerce** → 3 tests (real e-commerce)
- **The Internet** → 3 tests (UI elements + navigation)

### By Test Type
- **Form Filling** → 4 tests
- **E-Commerce** → 5 tests
- **UI Interactions** → 3 tests
- **Navigation** → 1 test

### By Difficulty
- **Easy** → 3 tests
- **Medium** → 4 tests
- **Hard** → 2 tests
- **Advanced** → 2 tests

---

## Website Characteristics

### Formy
✓ No authentication needed  
✓ Clean form layouts  
✓ Various input types  
✓ Great for learning form automation  

### Sauce Demo
✓ Multiple test users  
✓ Realistic e-commerce flow  
✓ Well-known in QA community  
✓ Stable and reliable  

### NopCommerce
✓ Real e-commerce platform  
✓ Complex product hierarchy  
✓ Realistic user workflows  
✓ Multi-step checkout  

### The Internet
✓ Diverse UI elements  
✓ No authentication  
✓ Great for edge case testing  
✓ Window/tab handling practice  

---

## How Tests Fail & Learn

### Common Failure Scenarios
1. **Form Tests** - Element identification, value injection, validation
2. **E-Commerce Tests** - Navigation, cart state, checkout flow
3. **UI Tests** - Dropdown/select handling, dynamic elements
4. **Multi-Window** - Context switching, new window detection

### Learning Outcomes
- Form field types and their handling
- Real vs. practice e-commerce patterns
- Dropdown/select element automation
- Complex multi-step workflows
- Error handling and recovery
- Performance under stress

---

## Next Steps to Add More Tests

You can easily add more tests with:

1. **More Form Sites:**
   - https://practice.automationcloud.net/
   - https://www.w3schools.com/

2. **More E-Commerce:**
   - https://www.opencart.com/ (demo)
   - https://prestashop-demo.com/

3. **More UI Element Tests:**
   - https://www.selenium.dev/selenium/web/tables.html
   - https://www.selenium.dev/selenium/web/alerts.html

4. **More Real Websites:**
   - GitHub (search + navigation)
   - Gmail (lightweight inbox)
   - Google Sheets (multi-element)

---

## Quick Reference

**For Quick Testing:** Use **Test Suite** tab
- 6 standardized tests
- One-click execution
- Fast feedback

**For Deep Learning:** Use **Testing Panel** tab
- 10 diverse scenarios
- Category filtering
- Detailed instructions
- Difficulty levels

